import numpy as np
import matplotlib.pyplot as plt
import sysid as sd

class Semp():

    @staticmethod
    def __plot(t, nu, ny, y, y_hat, title, error):
        plt.plot(t[:-max(nu, ny)], y[:-max(nu, ny)], label="Real")
        plt.plot(t[:-max(nu, ny)], y_hat, label="Prediction")
        plt.xlabel("time (s)")
        plt.title(title)
        plt.legend()
        plt.show()
        print('='*30)
        print(f'MSE of the Model = {error}')
        print('='*30)

    @staticmethod
    def __msse(y, y_hat):
        squared_errors = np.power(y - y_hat, 2)
        mse = np.mean(squared_errors)

        return mse

    @staticmethod
    def __prediction(psi_in, psi_out, y_train, i):
        aux = np.array(psi_in.copy())
        if i != 0:
            aux = np.column_stack((aux, psi_out[:,i]))

            theta = np.linalg.inv(aux.T @ aux) @ aux.T @ y_train

            y_hat = aux @ theta
        else:
            aux = psi_out[:,i].copy()

            a = np.dot(aux.T, aux)
            inv = 1/a
            theta = inv * aux.T @ y_train

            y_hat = aux * theta

        return y_hat, aux
    
    @staticmethod
    def __get_candidates(u_test, y_test, nu, ny, l, comb):
        cm_test = sd.candidate_matrix(sd.data_matrix(u_test, y_test, nu, ny), l)
        candidates_test = np.array([])
        for i in range(len(cm_test[1])):
            if cm_test[1][i] in comb:
                if len(candidates_test) == 0:
                    candidates_test = cm_test[0][:,i]
                else:
                    candidates_test = np.column_stack((candidates_test, cm_test[0][:,i]))

        return candidates_test

    @staticmethod
    def __run_semp(data_matrix, y_train, l, delay):
        y_train = y_train[:delay]
        psi_in = np.array([[]])
        comb_in = []
        psi_out, comb_out = sd.candidate_matrix(data_matrix, l)
        J = np.inf

        for i in range(psi_out.shape[1]):
            y_hat, aux = Semp.__prediction(psi_in, psi_out, y_train, i)
                
            Ji = Semp.__msse(y_train, y_hat)
            if Ji < J:
                J = Ji
                psi_in = aux.copy()
                comb_in.append(comb_out[i])
        
        theta = np.linalg.inv(psi_in.T @ psi_in) @ psi_in.T @ y_train
        y_hat = psi_in @ theta
        J = Semp.__msse(y_train, y_hat)

        i = 0
        while i < (psi_in.shape[1]):
            aux = psi_in.copy()
            aux = np.delete(aux, i, 1)
            theta = np.linalg.inv(aux.T @ aux) @ aux.T @ y_train
            y_hat = aux @ theta
            Ji = Semp.__msse(y_train, y_hat)
            if Ji < J:
                J = Ji
                psi_in = np.delete(psi_in, i, 1)
                comb_in.pop(i)
            else:
                i+=1

        return psi_in, comb_in
    
    @staticmethod
    def __run_train(u, y, nu, ny, l):
        data_matrix = sd.data_matrix(u, y, nu, ny)

        psi, comb = Semp.__run_semp(data_matrix, y, l, -max(nu, ny))

        theta = np.linalg.inv(psi.T @ psi) @ psi.T @ y[:-max(nu, ny)]
        y_hat = psi @ theta

        return y_hat, theta, comb

    def run(self, u, y, nu, ny, l, validation=0, title=''):
        t = np.arange(0, len(u) * 0.1, 0.1)

        if validation:
            n_train = int(0.2 * len(u))

            t_train, t_test = t[-n_train:], t[:-n_train]
            u_train, u_test = u[-n_train:], u[:-n_train]
            y_train, y_test = y[-n_train:], y[:-n_train]

            y_hat, theta, comb = Semp.__run_train(u_train, y_train, nu, ny, l)
            y_hat_test = Semp.__get_candidates(u_test, y_test, nu, ny, l, comb) @ theta

            Semp.__plot(t_train, nu, ny, y_train, y_hat, title+' - Train', Semp.__msse(y_train[:-max(nu, ny)], y_hat))
            Semp.__plot(t_test, nu, ny, y_test, y_hat_test, title+' - Validation', Semp.__msse(y_test[:-max(nu, ny)], y_hat_test))
            
            print('='*30)
            print(f'Non linear Regressors:')
            for i in range(len(comb)):
                print(sd.get_model_term(comb[i], nu, ny))
            print('='*30)

            self.y_hat_test = y_hat_test

        else:
            y_hat, theta, comb = Semp.__run_train(u, y, nu, ny, l)
            
            Semp.__plot(t, nu, ny, y, y_hat, title, Semp.__msse(y[:-max(nu, ny)], y_hat))

            print('='*30)
            print(f'Non linear Regressors:')
            for i in range(len(comb)):
                print(sd.get_model_term(comb[i], nu, ny))
            print('='*30)

        self.y_hat = y_hat
        self.theta = theta
        self.regressors = comb
