import numpy as np
import csv
import os

def field_generation(realization, operating_system):
    
    if operating_system == 'mac':
        os.system("chmod u+x hydrogen_mac")
    iteration = 0
    
    inputtxt = open('hydrogen_input.txt', "r")
    list_of_lines = inputtxt.readlines()
    randomseed = int(list_of_lines[0].split(" ")[0])
    nx = int(float(list_of_lines[3].split(" ")[0])) + 1
    ny = int(float(list_of_lines[3].split(" ")[1])) + 1
    mu = float(list_of_lines[7].split(" ")[0])
    sigma = float(list_of_lines[6].split(" ")[0])
    fields = np.zeros((realization,ny,nx))
    
    while iteration < realization:
        print(f'generating a field ...')
        modified = f'{randomseed}'
        modified += (18 - len(modified))*' ' + '! nseed (if set to 0 the seed is generated as random, otherwise the integer provided is used as seed)\n'
        list_of_lines[0] = modified
        a_file = open("hydrogen_input.txt", "w")
        a_file.writelines(list_of_lines)
        a_file.close()
        
        if operating_system == 'mac':
            os.system("./hydrogen_mac")
        if operating_system == 'linux':
            os.system("./hydrogen_linux")

        aquifer = []
        record = False

        with open('single_result.txt', newline='') as f:
            reader = csv.reader(f, skipinitialspace=True, delimiter=' ', quoting=csv.QUOTE_NONE)
            for row in reader:
                if record:
                    aquifer.append(row)
                if row[0] == 'replicate':
                    record = True        

        hcfield = []
        for i in aquifer:
            for j in i:
                hcfield.append(j)
        hcfield = np.asarray(hcfield, dtype='float')

        if ( ((mu*0.9) < hcfield.mean() and hcfield.mean() < (mu*1.1)) and
            ((sigma*0.9) < hcfield.var() and hcfield.var() < (sigma*1.1)) ):
            fields[iteration] = hcfield.reshape((ny,nx))
            print(f'Realization No. {iteration} done')
            iteration += 1
        randomseed += 1
    
    return fields
