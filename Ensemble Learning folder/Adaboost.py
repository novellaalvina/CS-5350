from itertools import count
import DT_EL as dt
import math
import numpy as np

# adaboost
def adaboost(dataset, T, columns, attributes,labels):
    
    # initialise weight for all examples
    D_t = [1/len(dataset) for ex in dataset]
    H = []
    votes = []
    H_final = []

    if len(T) > 0: 
        for t in T:

            # find h_t classifier whose weighted classifier error is better than chance(=sum(D_t(i)))
            h = dt.ID3(dataset, columns, attributes, labels, D_t, dt.GI(dt.calc_prob(dataset,labels)), max_depth=1,current_depth=0)
            H.append(h)

            # compute alpha_t(vote) 
            err = error_t(dataset, h, columns, D_t)
            alpha_t = math.log((1-err)/err)/2
            votes.append(alpha_t)

            # update D_t value for the training example, i.o.w, compute D_(t+1)
            i = 0
            for row in dataset:
                # h(x_i)
                h_x = dt.predict(row,h,columns)

                ### update weight ###
                # if y_i == h(x_i), then y_i * h(x_i) = 1
                if row[-1] == h_x:
                    D_t[i] *= math.exp(-alpha_t) 
                # if y_i != h(x_i), then y_i * h(x_i) = -1
                elif row[-1] != h_x:
                    D_t[i] *= math.exp(alpha_t) 
            
            # normalisation constant, z_t
            z_t = sum(D_t)
            D_t = [d/z_t for d in D_t] 
            ### update weight ends ###

            # h_final = sum(alpha_t*h_t(x)
            h_final = sum(alpha_t * h)
            if h_final >=0:
                H_final.append(1)
            elif h_final<0:
                H_final.append(-1)
        
        # sgn(h_final)
        counter = {1:0, -1:0}
        for h_f in H_final:
            if h_f == 1:
                counter[1] += 1
            else:
                counter[-1] += 1
        
        return max(counter, key=counter.get)

# compute err_t = true error
def error_t(dataset, h, columns, D_t):
    D_incorrect = []
    i = 0
    for row in dataset:
        # h(x_i)
        y_predict = dt.predict(dataset,h,columns)

        # D_t if y_i != h(x_i)
        if row[-1] != y_predict:
            D_incorrect.append(D_t[i])
    
    return sum(D_incorrect)/len(dataset)
