import requests
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd

from Data import data_rates, std, Pearson_cor, data_closing_prices, rates_for_backtest, closing_for_backtest, rates_for_bootstrap, closing_for_bootstrap


def Mod_Sharpe_Ratio(allocation_sp500, allocation_tsx):
    allocation_ex_us_can = 100 - allocation_sp500 - allocation_tsx
    ponderation_sp500 = allocation_sp500 / 100
    ponderation_tsx = allocation_tsx / 100
    ponderation_ex_us_can = allocation_ex_us_can / 100

    #we will use the geometric average return, since we are compounding the returns on a monthly basis.
    average_sp500_monthly = 1
    for i in range(len(rates_for_backtest["SP500"])):
        average_sp500_monthly *= (1 + rates_for_backtest["SP500"][i][1])
    average_sp500_monthly = average_sp500_monthly**(1/len(rates_for_backtest["SP500"])) - 1

    average_tsx_monthly = 1
    for i in range(len(rates_for_backtest["TSX"])):
        average_tsx_monthly *= (1 + rates_for_backtest["TSX"][i][1])
    average_tsx_monthly = average_tsx_monthly**(1/len(rates_for_backtest["TSX"])) - 1

    average_ex_us_can_monthly = 1
    for i in range(len(rates_for_backtest["Ex_US_Can"])):
        average_ex_us_can_monthly *= (1 + rates_for_backtest["Ex_US_Can"][i][1])
    average_ex_us_can_monthly = average_ex_us_can_monthly**(1/len(rates_for_backtest["Ex_US_Can"])) - 1

    return_rate_monthly = ponderation_sp500 * average_sp500_monthly + \
                           ponderation_tsx * average_tsx_monthly + \
                            ponderation_ex_us_can * average_ex_us_can_monthly
    #We will use the sharpe ratio in a way that the excess return is using the cost of the loan and not the risk free rate.
    #The cost of the loan is T+0.005
    #Where T is the preferential rate of banks (policy rate + 0.022)
    average_policy_rate_monthly = np.mean([rates_for_backtest["Policy_Rate_BOC"][i][1] for i in range(len(rates_for_backtest["Policy_Rate_BOC"]))])
    policy_rate_annual = (1 + average_policy_rate_monthly)**12 - 1
    marge_annual = policy_rate_annual + 0.022 + 0.005
    cost_of_loan_monthly = (1 + marge_annual)**(1/12) - 1

    #We will use the excess return on a mountly basis, since the std is mounthly.
    excess_returns = return_rate_monthly - cost_of_loan_monthly
    #portfolio variance formula of a 3 asset portfolio with correlation (Modern Portfolio Theory, Markowitz)
    portefolio_variance = ponderation_sp500**2 * std["SP500"]**2 + \
                          ponderation_tsx**2 * std["TSX"]**2 + \
                            ponderation_ex_us_can**2 * std["Ex_US_Can"]**2 + \
                            2 * ponderation_sp500 * ponderation_tsx * std["SP500"] * std["TSX"] * Pearson_cor["SP500_TSX"] + \
                            2 * ponderation_sp500 * ponderation_ex_us_can * std["SP500"] * std["Ex_US_Can"] * Pearson_cor["SP500_Ex_US_Can"] + \
                            2 * ponderation_tsx * ponderation_ex_us_can * std["TSX"] * std["Ex_US_Can"] * Pearson_cor["TSX_Ex_US_Can"]

    sharpe_ratio = excess_returns / (portefolio_variance)**0.5 if portefolio_variance != 0 else 0

    return sharpe_ratio * np.sqrt(12)

def find_optimal_allocation():
    optimal_allocation = None
    max_sharpe_ratio = float('-inf')

    for allocation_sp500 in range(0, 101, 5):
        for allocation_tsx in range(0, 100 - allocation_sp500 + 1, 5):
            sharpe_ratio = Mod_Sharpe_Ratio(allocation_sp500, allocation_tsx)
            if sharpe_ratio > max_sharpe_ratio or optimal_allocation is None:
                max_sharpe_ratio = sharpe_ratio
                optimal_allocation = (allocation_sp500, allocation_tsx, 100 - allocation_sp500 - allocation_tsx)

    return f"Optimal Allocation: SP500: {optimal_allocation[0]}%, TSX: {optimal_allocation[1]}%, Ex_US_Can: {optimal_allocation[2]}% \n With a Sharpe Ratio of {max_sharpe_ratio:.4f}"

def backtest_portfolio(allocation_sp500, allocation_tsx, years):
    allocation_ex_us_can = 100 - allocation_sp500 - allocation_tsx
    ponderation_sp500 = allocation_sp500 / 100
    ponderation_tsx = allocation_tsx / 100
    ponderation_ex_us_can = allocation_ex_us_can / 100

    number_of_months = years * 12

    output = []

    for i in range(len(closing_for_backtest["SP500"]) - number_of_months):
        Growth_sp500 = (closing_for_backtest["SP500"][i+number_of_months][1] / closing_for_backtest["SP500"][i][1]) -1
        Growth_tsx = (closing_for_backtest["TSX"][i+number_of_months][1] / closing_for_backtest["TSX"][i][1]) -1
        Growth_ex_us_can = (closing_for_backtest["Ex_US_Can"][i+number_of_months][1] / closing_for_backtest["Ex_US_Can"][i][1]) -1
        portfolio_end_value = 1000 * (1 + ponderation_sp500 * Growth_sp500 +\
                                   ponderation_tsx * Growth_tsx + ponderation_ex_us_can * Growth_ex_us_can)
        loan = 1000
        position_SP = 0
        position_TSX = 0
        position_Ex_US_Can = 0

        #working capital/ 12 mouth of cashflow
        working_capital_1_year = 12 * loan * ((1 + (data_rates["Policy_Rate_BOC"][i][1] + 0.022 + 0.005))**(1/12) - 1)
        position_SP += working_capital_1_year * ponderation_sp500 / closing_for_backtest["SP500"][i][1]
        position_TSX += working_capital_1_year * ponderation_tsx / closing_for_backtest["TSX"][i][1]
        position_Ex_US_Can += working_capital_1_year * ponderation_ex_us_can / closing_for_backtest["Ex_US_Can"][i][1]
        #The interest only 3 year period
        for j in range(0, 36):
            interest = loan * ((1 + (data_rates["Policy_Rate_BOC"][i+j][1] + 0.022 + 0.005))**(1/12) - 1)
            position_SP += interest * ponderation_sp500 / closing_for_backtest["SP500"][i+j][1]
            position_TSX += interest * ponderation_tsx / closing_for_backtest["TSX"][i+j][1]
            position_Ex_US_Can += interest * ponderation_ex_us_can / closing_for_backtest["Ex_US_Can"][i+j][1]

        #increase in working capital/ 12 mouth of cashflow
        mounthly_rate = (1 + (data_rates["Policy_Rate_BOC"][i+36][1] + 0.022 + 0.005))**(1/12) - 1
        total_working_capital = loan * mounthly_rate / (1 - (1 + mounthly_rate)**(36 - number_of_months))
        working_capital_increase = total_working_capital - working_capital_1_year
        position_SP += working_capital_increase * ponderation_sp500 / closing_for_backtest["SP500"][i+36][1]
        position_TSX += working_capital_increase * ponderation_tsx / closing_for_backtest["TSX"][i+36][1]
        position_Ex_US_Can += working_capital_increase * ponderation_ex_us_can / closing_for_backtest["Ex_US_Can"][i+36][1]
        for j in range(36, number_of_months):
            mounthly_rate = (1 + (data_rates["Policy_Rate_BOC"][i+j][1] + 0.022 + 0.005))**(1/12) - 1
            payement = loan * mounthly_rate / (1 - (1 + mounthly_rate)**(j- number_of_months))
            interest = loan * mounthly_rate
            loan += interest - payement
            position_SP += payement * ponderation_sp500 / closing_for_backtest["SP500"][i+j][1]
            position_TSX += payement * ponderation_tsx / closing_for_backtest["TSX"][i+j][1]
            position_Ex_US_Can += payement * ponderation_ex_us_can / closing_for_backtest["Ex_US_Can"][i+j][1]

        # removing the working capital
        position_SP -= working_capital_1_year * ponderation_sp500 / closing_for_backtest["SP500"][i+number_of_months][1]
        position_TSX -= working_capital_1_year * ponderation_tsx / closing_for_backtest["TSX"][i+number_of_months][1]
        position_Ex_US_Can -= working_capital_1_year * ponderation_ex_us_can / closing_for_backtest["Ex_US_Can"][i+number_of_months][1]

        comparation_end_value = position_SP * closing_for_backtest["SP500"][i+number_of_months][1] +\
                               position_TSX * closing_for_backtest["TSX"][i+number_of_months][1] +\
                               position_Ex_US_Can * closing_for_backtest["Ex_US_Can"][i+number_of_months][1]
        output.append((closing_for_backtest["SP500"][i][0], portfolio_end_value, comparation_end_value, portfolio_end_value - comparation_end_value, loan))

    return output

def bootstrap(means_global, matrix_cov_global):
    len_rates = len(rates_for_bootstrap["SP500"])
    choose_time_step = np.random.randint(0, len_rates)
    return (rates_for_bootstrap["SP500"][choose_time_step][1], rates_for_bootstrap["TSX"][choose_time_step][1], rates_for_bootstrap["Ex_US_Can"][choose_time_step][1])

def normal(means_global, matrix_cov_global):
    # Génère uniquement l'échantillon en utilisant la matrice déjà calculée
    rate_sp500, rate_tsx, rate_ex = np.random.multivariate_normal(means_global, matrix_cov_global)
    return (rate_sp500, rate_tsx, rate_ex)

def student(means_global, matrix_cov_global, df=5):
    # Ajustement de la matrice de covariance pour Student
    matrix_dispersion = matrix_cov_global * ((df - 2) / df)
    
    z = np.random.multivariate_normal(np.zeros(3), matrix_dispersion)
    u = np.random.chisquare(df)
    
    rate_sp500, rate_tsx, rate_ex = means_global + z / np.sqrt(u / df)
    return (rate_sp500, rate_tsx, rate_ex)

def montecarlo(allocation_sp500, allocation_tsx, years, iterations, loan_rate, change_rate_fonction):
    allocation_ex_us_can = 100 - allocation_sp500 - allocation_tsx
    ponderation_sp500 = allocation_sp500 / 100
    ponderation_tsx = allocation_tsx / 100
    ponderation_ex_us_can = allocation_ex_us_can / 100
    number_of_months = years * 12

    #To make the montecarlo more efficient, we will generate all the random rates at once and then use them in the simulation. 
    # This way we avoid generating random numbers in the inner loop, which can be computationally expensive.
    ret_sp500 = [x[1] for x in rates_for_bootstrap["SP500"]]
    ret_tsx = [x[1] for x in rates_for_bootstrap["TSX"]]
    ret_ex = [x[1] for x in rates_for_bootstrap["Ex_US_Can"]]
    means_global = np.array([np.mean(ret_sp500), np.mean(ret_tsx), np.mean(ret_ex)])
    data_matrix_global = np.array([ret_sp500, ret_tsx, ret_ex])
    matrix_cov_global = np.cov(data_matrix_global)
    output = []
    for _ in range(iterations):
        loan = 1000
        current_sp_price = closing_for_bootstrap["SP500"][-1][1]
        current_tsx_price = closing_for_bootstrap["TSX"][-1][1]
        current_ex_us_can_price = closing_for_bootstrap["Ex_US_Can"][-1][1]
        position_lump_sum_sp500 = loan * ponderation_sp500 / current_sp_price
        position_lump_sum_tsx = loan * ponderation_tsx / current_tsx_price
        position_lump_sum_ex_us_can = loan * ponderation_ex_us_can / current_ex_us_can_price
        mounthly_rate = (1 + loan_rate)**(1/12) - 1
        #working capital/ 12 mouth of cashflow
        position_sp500 = loan * loan_rate * ponderation_sp500 / current_sp_price
        position_tsx = loan * loan_rate * ponderation_tsx / current_tsx_price
        position_ex_us_can = loan * loan_rate * ponderation_ex_us_can / current_ex_us_can_price
        for j in range(0, 36):
            #price change
            price_change = change_rate_fonction(means_global, matrix_cov_global)
            current_sp_price *= (1 + price_change[0])
            current_tsx_price *= (1 + price_change[1])
            current_ex_us_can_price *= (1 + price_change[2])

            interest = loan * mounthly_rate
            position_sp500 += interest * ponderation_sp500 / current_sp_price
            position_tsx += interest * ponderation_tsx / current_tsx_price
            position_ex_us_can += interest * ponderation_ex_us_can / current_ex_us_can_price

        #increase in working capital/ 12 mouth of cashflow
        total_working_capital = loan * mounthly_rate / (1 - (1 + mounthly_rate)**(36 - number_of_months))
        working_capital_increase = total_working_capital - loan * loan_rate
        position_sp500 += working_capital_increase * ponderation_sp500 / current_sp_price
        position_tsx += working_capital_increase * ponderation_tsx / current_tsx_price
        position_ex_us_can += working_capital_increase * ponderation_ex_us_can / current_ex_us_can_price
        for j in range(36, number_of_months):
            #price change
            price_change = change_rate_fonction(means_global, matrix_cov_global)
            current_sp_price *= (1 + price_change[0])
            current_tsx_price *= (1 + price_change[1])
            current_ex_us_can_price *= (1 + price_change[2])

            payement = loan * mounthly_rate / (1 - (1 + mounthly_rate)**(j- number_of_months))
            interest = loan * mounthly_rate
            loan += interest - payement
            position_sp500 += payement * ponderation_sp500 / current_sp_price
            position_tsx += payement * ponderation_tsx / current_tsx_price
            position_ex_us_can += payement * ponderation_ex_us_can / current_ex_us_can_price
        # removing the working capital
        position_sp500 -= total_working_capital * ponderation_sp500 / current_sp_price
        position_tsx -= total_working_capital * ponderation_tsx / current_tsx_price
        position_ex_us_can -= total_working_capital * ponderation_ex_us_can / current_ex_us_can_price
        portfolio_end_value_dca = position_sp500 * current_sp_price + position_tsx * current_tsx_price + position_ex_us_can * current_ex_us_can_price
        portfolio_end_value_lump_sum = position_lump_sum_sp500 * current_sp_price + position_lump_sum_tsx * current_tsx_price + position_lump_sum_ex_us_can * current_ex_us_can_price
        comparation_end_value = portfolio_end_value_lump_sum - portfolio_end_value_dca
        output.append(("No date in this context", portfolio_end_value_lump_sum, portfolio_end_value_dca, comparation_end_value, loan))
    return output

def exit_3_years(allocation_sp500, allocation_tsx, iterations, loan_rate, change_rate_fonction):
    years = 3
    allocation_ex_us_can = 100 - allocation_sp500 - allocation_tsx
    ponderation_sp500 = allocation_sp500 / 100
    ponderation_tsx = allocation_tsx / 100
    ponderation_ex_us_can = allocation_ex_us_can / 100
    number_of_months = years * 12

    #To make the montecarlo more efficient, we will generate all the random rates at once and then use them in the simulation. 
    # This way we avoid generating random numbers in the inner loop, which can be computationally expensive.
    ret_sp500 = [x[1] for x in rates_for_bootstrap["SP500"]]
    ret_tsx = [x[1] for x in rates_for_bootstrap["TSX"]]
    ret_ex = [x[1] for x in rates_for_bootstrap["Ex_US_Can"]]
    means_global = np.array([np.mean(ret_sp500), np.mean(ret_tsx), np.mean(ret_ex)])
    data_matrix_global = np.array([ret_sp500, ret_tsx, ret_ex])
    matrix_cov_global = np.cov(data_matrix_global)
    output = []
    for _ in range(iterations):
        loan = 1000
        current_sp_price = closing_for_bootstrap["SP500"][-1][1]
        current_tsx_price = closing_for_bootstrap["TSX"][-1][1]
        current_ex_us_can_price = closing_for_bootstrap["Ex_US_Can"][-1][1]
        position_lump_sum_sp500 = loan * ponderation_sp500 / current_sp_price
        position_lump_sum_tsx = loan * ponderation_tsx / current_tsx_price
        position_lump_sum_ex_us_can = loan * ponderation_ex_us_can / current_ex_us_can_price
        mounthly_rate = (1 + loan_rate)**(1/12) - 1
        #working capital/ 12 mouth of cashflow
        position_sp500 = loan * loan_rate * ponderation_sp500 / current_sp_price
        position_tsx = loan * loan_rate * ponderation_tsx / current_tsx_price
        position_ex_us_can = loan * loan_rate * ponderation_ex_us_can / current_ex_us_can_price
        for j in range(0, 36):
            #price change
            price_change = change_rate_fonction(means_global, matrix_cov_global)
            current_sp_price *= (1 + price_change[0])
            current_tsx_price *= (1 + price_change[1])
            current_ex_us_can_price *= (1 + price_change[2])

            interest = loan * mounthly_rate
            position_sp500 += interest * ponderation_sp500 / current_sp_price
            position_tsx += interest * ponderation_tsx / current_tsx_price
            position_ex_us_can += interest * ponderation_ex_us_can / current_ex_us_can_price
        # removing the working capital
        position_sp500 -= loan * loan_rate * ponderation_sp500 / current_sp_price
        position_tsx -= loan * loan_rate * ponderation_tsx / current_tsx_price
        position_ex_us_can -= loan * loan_rate * ponderation_ex_us_can / current_ex_us_can_price
        portfolio_end_value_dca = position_sp500 * current_sp_price + position_tsx * current_tsx_price + position_ex_us_can * current_ex_us_can_price
        portfolio_end_value_lump_sum = position_lump_sum_sp500 * current_sp_price + position_lump_sum_tsx * current_tsx_price + position_lump_sum_ex_us_can * current_ex_us_can_price
        comparation_end_value = portfolio_end_value_lump_sum - portfolio_end_value_dca - loan
        output.append(("No date in this context", portfolio_end_value_lump_sum, portfolio_end_value_dca, comparation_end_value, loan))
    return output

def plot_backtest_results(results):
    dates = pd.to_datetime([result[0] for result in results])

    plt.figure(figsize=(12, 6))
    plt.plot(dates, [result[3] for result in results], label='Difference (Portfolio - Comparison)', color='green')
    plt.xlabel('Date of start')
    plt.ylabel('Value ($)')
    plt.title('Backtest Results: Compound Portfolio - Compound Cashflow Comparison')
    plt.legend()
    plt.grid()
    plt.show()

def dispertion_of_results(results, bin, name):
    differences = [result[3] for result in results]
    print(f"Results Analysis for {name}:")
    print(f"Number of Simulations: {len(differences)}")
    print(f"Average Difference: {np.mean(differences):.2f}")
    print(f"Percentage of Positive Outcomes: {np.mean([1 if diff > 0 else 0 for diff in differences]) * 100:.2f}%")
    print(f"Maximum Difference: {np.max(differences):.2f}")
    print(f"Minimum Difference: {np.min(differences):.2f}")
    print(f"5% Quantile: {np.percentile(differences, 5):.2f}")
    print(f"95% Quantile: {np.percentile(differences, 95):.2f}")
    plt.figure(figsize=(10, 5))
    plt.hist(differences, bins=bin, color='b')
    plt.xlabel('Difference (Portfolio Value - Comparison Value)')
    plt.ylabel('Frequency')
    plt.title('Dispersion of Portfolio Performance Compared to its DCA cost using ' + name)
    plt.grid()
    plt.show()

def cashflow(loan, rate_of_loan, years):
    initial_loan = loan
    months = years * 12
    #we want an output of mounthly cashflow of the 3 years buffer and of the repayement period
    mounthly_rate = (1 + rate_of_loan)**(1/12) - 1
    cashflow_list = []
    for j in range(0, 36):
        interest = loan * mounthly_rate
        cashflow_list.append(interest)
    for j in range(36, months):
        payement = loan * mounthly_rate / (1 - (1 + mounthly_rate)**(j- months))
        interest = loan * mounthly_rate
        loan += interest - payement
        cashflow_list.append(payement)
    output = f"For a loan of {initial_loan} with a rate of {round(rate_of_loan*100, 2)}% and a repayement period of {years - 3} years: \n \n"
    output += "First 3 years of interest only cashflow: \n"
    output += "Mounthly cashflow: " + str(cashflow_list[0]) + "\n"
    output += "Yearly cashflow: " + str(cashflow_list[0] * 12) + "\n" + "\n"
    output += "Repayement period cashflow: \n"
    output += "Mounthly cashflow: " + str(cashflow_list[36]) + "\n"
    output += "Yearly cashflow: " + str(cashflow_list[36] * 12) + "\n" +  "\n" + "\n"
    return output