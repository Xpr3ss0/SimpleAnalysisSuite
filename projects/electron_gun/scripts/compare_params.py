

# compare conent from different parmeter files by comparing them line by line and printing out the differences

base_path = r"D:\VSCode Projects\SimpleAnalysisSuite\projects\electron_gun\data\17.10.2025\controlled_comparison\param_set_"

param_files = [base_path + f"{i}.ini" for i in range(1, 5)]

param_dicts = []
for pf in param_files:
    params = {}
    with open(pf, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()
    param_dicts.append(params)

# compare all sets to set 1

for i in range(1, len(param_dicts)):
    print(f"Comparing param_set_1.ini to param_set_{i+1}.ini")
    params1 = param_dicts[0]
    params2 = param_dicts[i]
    all_keys = set(params1.keys()).union(set(params2.keys()))
    for key in all_keys:
        val1 = params1.get(key, "<MISSING>")
        val2 = params2.get(key, "<MISSING>")
        if val1 != val2:
            print(f"  Difference in '{key}': '{val1}' vs '{val2}'")
    print()