f = open("submit.csv")
g = open("submit2.csv", "w")
g.write("Id,Category\n")
r = f.readlines()
for i in range(4977): 
    s = 0
    #print('here')
    for line in r:
        #print("here")
        try:
            m = line.split(',')
        except: 
            m = 'str'
        try:
            u = int(m[0])
        except:
            u = -1
        if u == i:
            #print("Caught")
            g.write(line)
            r.remove(line)
            break
    
            
    