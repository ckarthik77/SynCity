import random, csv
INPUT="telemetry.csv"
OUT="telemetry_reservoir.csv"
K=100000

reservoir=[]
with open(INPUT,'r',newline='') as f:
    reader=csv.reader(f)
    header=next(reader)
    for i,row in enumerate(reader):
        if i < K:
            reservoir.append(row)
        else:
            j=random.randint(0,i)
            if j < K:
                reservoir[j]=row

with open(OUT,'w',newline='') as f:
    writer=csv.writer(f)
    writer.writerow(header)
    writer.writerows(reservoir)
print("Reservoir done. Sample size:", len(reservoir))
