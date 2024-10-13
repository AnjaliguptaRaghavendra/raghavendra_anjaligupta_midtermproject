#!/usr/bin/env python
# coding: utf-8

# # CS634101 MIDTERM PROJECT

# Name: Anjaligupta Raghavendra

# UCID: ar2729

# PROFESSOR: Dr.Yasser Abduallah

# In[20]:


import pandas as pd


# In[22]:


import itertools
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
import time

# Convert transactions into one-hot encoded dataframe for Apriori and FP-Growth
def encode_transactions(transactions):
    unique_items = set(item for sublist in transactions for item in sublist)
    encoded_vals = []
    for transaction in transactions:
        encoded_vals.append({item: (item in transaction) for item in unique_items})
    return pd.DataFrame(encoded_vals)

# Loading transactions from CSV files
def load_transactions_from_csv(file_path):
    df = pd.read_csv(file_path)
    transactions = df['Transaction'].apply(lambda x: x.split(',')).tolist()
    return transactions

# Brute-Force Algorithm

#Finding Itemsets
def brute_force_frequent_itemsets(transactions, min_support):
    unique_items = set(item for sublist in transactions for item in sublist)
    n_transactions = len(transactions)
    frequent_itemsets = []

    for size in range(1, len(unique_items) + 1):
        for itemset in itertools.combinations(unique_items, size):
            count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
            support = count / n_transactions * 100
            if support >= min_support:
                frequent_itemsets.append((itemset, support))

    return frequent_itemsets

#Association Rules
def brute_force_association_rules(frequent_itemsets, transactions, min_confidence):
    n_transactions = len(transactions)
    rules = []

    for itemset, support in frequent_itemsets:
        if len(itemset) < 2:
            continue
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                antecedent_count = sum(1 for transaction in transactions if set(antecedent).issubset(set(transaction)))
                confidence = support / (antecedent_count / n_transactions * 100) * 100
                if confidence >= min_confidence:
                    rules.append((antecedent, consequent, support, confidence))

    return rules

def format_brute_force_output(frequent_itemsets, rules):
    print("\nBrute Force Frequent Itemsets:")
    for itemset, support in frequent_itemsets:
        print(f"Itemset: {itemset}, Support: {support:.2f}%")

    print("\nBrute Force Association Rules:")
    for antecedent, consequent, support, confidence in rules:
        print(f"Rule: {antecedent} -> {consequent}, Support: {support:.2f}%, Confidence: {confidence:.2f}%")

# Apriori Algorithm

def run_apriori_algorithm(transactions, min_support, min_confidence):
    df_encoded = encode_transactions(transactions)
    frequent_itemsets = apriori(df_encoded, min_support=min_support/100, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence/100)
    return frequent_itemsets, rules

def format_apriori_output(frequent_itemsets, rules):
    print("\nApriori Frequent Itemsets:")
    for idx, row in frequent_itemsets.iterrows():
        itemset = tuple(row['itemsets'])
        support = row['support'] * 100
        print(f"Itemset: {itemset}, Support: {support:.2f}%")
    
    print("\nApriori Association Rules:")
    for idx, row in rules.iterrows():
        antecedent = tuple(row['antecedents'])
        consequent = tuple(row['consequents'])
        support = row['support'] * 100
        confidence = row['confidence'] * 100
        print(f"Rule: {antecedent} -> {consequent}, Support: {support:.2f}%, Confidence: {confidence:.2f}%")

#FP-Growth Algorithm

def run_fpgrowth_algorithm(transactions, min_support, min_confidence):
    df_encoded = encode_transactions(transactions)
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support/100, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence/100)
    return frequent_itemsets, rules

def format_fpgrowth_output(frequent_itemsets, rules):
    print("\nFP-Growth Frequent Itemsets:")
    for idx, row in frequent_itemsets.iterrows():
        itemset = tuple(row['itemsets'])
        support = row['support'] * 100
        print(f"Itemset: {itemset}, Support: {support:.2f}%")
    
    print("\nFP-Growth Association Rules:")
    for idx, row in rules.iterrows():
        antecedent = tuple(row['antecedents'])
        consequent = tuple(row['consequents'])
        support = row['support'] * 100
        confidence = row['confidence'] * 100
        print(f"Rule: {antecedent} -> {consequent}, Support: {support:.2f}%, Confidence: {confidence:.2f}%")

# Compare results between algorithms
def compare_results(bf_rules, apriori_rules, fpgrowth_rules):
    bf_set = set((tuple(r[0]), tuple(r[1])) for r in bf_rules)
    apriori_set = set((tuple(r['antecedents']), tuple(r['consequents'])) for _, r in apriori_rules.iterrows())
    fpgrowth_set = set((tuple(r['antecedents']), tuple(r['consequents'])) for _, r in fpgrowth_rules.iterrows())

    print("\nAre the association rules the same between the algorithms?")
    print("Brute-Force vs Apriori: ", bf_set == apriori_set)
    print("Brute-Force vs FP-Growth: ", bf_set == fpgrowth_set)
    print("Apriori vs FP-Growth: ", apriori_set == fpgrowth_set)

# fastest algorithm
def display_fastest_algorithm(brute_force_time, apriori_time, fpgrowth_time):
    if brute_force_time < apriori_time and brute_force_time < fpgrowth_time:
        fastest_algorithm = "Brute Force"
        fastest_time = brute_force_time
    elif apriori_time < brute_force_time and apriori_time < fpgrowth_time:
        fastest_algorithm = "Apriori"
        fastest_time = apriori_time
    else:
        fastest_algorithm = "FP-Growth"
        fastest_time = fpgrowth_time

    print(f"\nFastest Algorithm: {fastest_algorithm}")
    print(f"Execution Time: {fastest_time:.4f} seconds")

# Main
def compare_algorithms(store, min_support, min_confidence):
    if store == 1:
        file_path = "amazon.csv"
    elif store == 2:
        file_path = "bestbuy.csv"
    elif store == 3:
        file_path = "kmart.csv"
    elif store == 4:
        file_path = "nike.csv"
    elif store == 5:
        file_path = "generic.csv"
    else:
        print("Invalid store selection.")
        return
    
    transactions = load_transactions_from_csv(file_path)
    
    #Brute Force Itemsets
    bf_frequent_itemsets = brute_force_frequent_itemsets(transactions, min_support)
    bf_rules = brute_force_association_rules(bf_frequent_itemsets, transactions, min_confidence)
    format_brute_force_output(bf_frequent_itemsets, bf_rules)
    
    #Apriori Itemsets
    apriori_frequent_itemsets, apriori_rules = run_apriori_algorithm(transactions, min_support, min_confidence)
    format_apriori_output(apriori_frequent_itemsets, apriori_rules)
    
    #FP-Growth Itemsets
    fpgrowth_frequent_itemsets, fpgrowth_rules = run_fpgrowth_algorithm(transactions, min_support, min_confidence)
    format_fpgrowth_output(fpgrowth_frequent_itemsets, fpgrowth_rules)

    #TIME- Brute Force
    start_time = time.time()
    bf_frequent_itemsets = brute_force_frequent_itemsets(transactions, min_support)
    bf_rules = brute_force_association_rules(bf_frequent_itemsets, transactions, min_confidence)
    brute_force_time = time.time() - start_time
    print(f"\nBrute Force Execution Time: {brute_force_time:.4f} seconds")

    #TIME- Apriori
    start_time = time.time()
    apriori_frequent_itemsets, apriori_rules = run_apriori_algorithm(transactions, min_support, min_confidence)
    apriori_time = time.time() - start_time
    print(f"Apriori Execution Time: {apriori_time:.4f} seconds")

    #TIME-  FP-Growth
    start_time = time.time()
    fpgrowth_frequent_itemsets, fpgrowth_rules = run_fpgrowth_algorithm(transactions, min_support, min_confidence)
    fpgrowth_time = time.time() - start_time
    print(f"FP-Growth Execution Time: {fpgrowth_time:.4f} seconds")

    #Displaying the fastest algorithm
    display_fastest_algorithm(brute_force_time, apriori_time, fpgrowth_time)

    #Comparing results
    compare_results(bf_rules, apriori_rules, fpgrowth_rules)

# Asking for user input
def main():
    while True:
        try:
            store = int(input("Choose the store-\n1. Amazon\n2. Best Buy\n3. K-mart\n4. Nike\n5. Generic\n"))
            if store < 1 or store > 5:
                raise ValueError
            break
        except ValueError:
            print("Invalid store selection. Please choose a number between 1 and 5.")

    while True:
        try:
            min_support = int(input("Enter the minimum support (1-100): "))
            if min_support < 1 or min_support > 100:
                raise ValueError
            break
        except ValueError:
            print("Invalid support value. Please enter a number between 1 and 100.")

    while True:
        try:
            min_confidence = int(input("Enter the minimum confidence (1-100): "))
            if min_confidence < 1 or min_confidence > 100:
                raise ValueError
            break
        except ValueError:
            print("Invalid confidence value. Please enter a number between 1 and 100.")
    
    compare_algorithms(store, min_support, min_confidence)

if __name__ == "__main__":
    main()


# In[ ]:




