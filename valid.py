import numpy as np

def validation(net, y):
    count = 0
    for i in range(len(y)):
        if net[i] == list(y[i]):
            count += 1
        else:
            pass
    return 100*count/len(y)

def validation_train(net, y, f=None):
    count = validation(net, y)
    if f == None:
        pass      
    else:
        print('\nTRAIN:', '\n', 'number of True set_train ', str(count), ' from ', str(len(y)), '\n', 'True in %: ', str(100*count/len(y)), file=f)
    print('\nTRAIN:')        
    print('number of True set_train', count, 'from', len(y))
    # print('True in %:', 100*count/len(y))
    print('True in %:', count)

def validation_test(net, y, f=None):
    count = validation(net, y)
    if f == None:
        pass  
    else:
        print('\nTEST:', '\n', 'number of True set_test ', str(count), ' from ', str(len(y)), '\n', 'True in %: ', str(100*count/len(y)), file=f)
            # print(net[i] == y[i])
    print('\nTEST:')
    print('number of True set_test', count, 'from', len(y))
    # print('True in %:', 100*count/len(y))
    print('True in %:', count)
    

def accuracy(obj, x_train, x_test, y_train, y_test, best):
    valid_list = []
    for best_one in best:
        output_net = []
        for i in range(len(y_train)):   
            output_net.append(obj.predict(x_train[i], best_one))

        output_tst = []
        for i in range(len(y_test)):
            output_tst.append(obj.predict(x_test[i], best_one))
        valid_list.append((validation(output_net, y_train), validation(output_tst, y_test)))
    return np.asarray(valid_list)
