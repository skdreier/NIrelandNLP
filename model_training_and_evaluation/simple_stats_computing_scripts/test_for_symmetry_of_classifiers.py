import sys


counts_csv = sys.argv[1]


model_1_name = None
model_2_name = None
model1_to_model2_to_count = {}
with open(counts_csv, 'r') as f:
    model_2_name = f.readline().strip().strip(',')
    number_of_categories = int(f.readline().strip().split(',')[-1]) + 1
    line_counter = 0
    for line in f:
        if not line.startswith(','):
            model_1_name = line[:line.index(',')]
        line = line[line.index(',') + 1:]
        line = line[line.index(',') + 1:]
        line = [int(piece) for piece in line.split(',')]
        assert len(line) == number_of_categories
        model1_to_model2_to_count[line_counter] = {}
        for i in range(number_of_categories):
            model1_to_model2_to_count[line_counter][i] = line[i]
        line_counter += 1


# now compute test statistic and dof
dof = number_of_categories * (number_of_categories - 1) // 2
total = 0
for j in range(number_of_categories):
    for i in range(j):
        cell1 = model1_to_model2_to_count[i][j]
        cell2 = model1_to_model2_to_count[j][i]
        if cell1 + cell2 > 0:
            total += ((cell2 - cell1) * (cell2 - cell1)) / (cell1 + cell2)
print('McNemar-Bowker test statistic: ' + str(total))
print('DoF: ' + str(dof))
