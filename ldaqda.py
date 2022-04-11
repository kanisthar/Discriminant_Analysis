import numpy as np
import matplotlib.pyplot as plt
import util

def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """

    cov = np.cov(x.T)

    #cov for female
    female_ind = np.where(np.column_stack((y,y)).ravel() == 2)[0]
    x_female = np.take(x.ravel(), female_ind)
    x_female = np.reshape(x_female, (-1, 2))
    cov_female = np.cov(x_female.T)

    #cov for male
    male_ind = np.where(np.column_stack((y,y)).ravel() == 1)[0]
    x_male = np.take(x.ravel(), male_ind)
    x_male = np.reshape(x_male, (-1,2))
    cov_male = np.cov(x_male.T)

    #mu for female (expectation)
    mu_female = np.sum(x_female, axis=0)/x_female.shape[0]

    #mu for male (expectation)
    mu_male = np.sum(x_male, axis=0)/x_male.shape[0]

    #defined parameters
    pi = 0.5
    height = x[:,0]
    width = x[:,1]

    plt.figure(1)

    #decision boundary calculations for LDA
    height_range = np.linspace(50, 80,100)
    width_range = np.linspace(80, 280, 100)
    h_1, w_1 = np.meshgrid(height_range, width_range)
    H_1 = h_1.ravel()
    W_1 = w_1.ravel()

    female_LDA_2 = util.density_Gaussian(mu_female, cov, np.column_stack((H_1, W_1)))
    male_LDA_2 = util.density_Gaussian(mu_male, cov, np.column_stack((H_1, W_1)))
    function = (male_LDA_2.T - female_LDA_2.T).reshape(h_1.shape)

    #plotting contours for LDA
    plt.contour(h_1, w_1, function, levels=[0])
    plt.contour(h_1, w_1, female_LDA_2.reshape(h_1.shape))
    plt.contour(h_1, w_1, male_LDA_2.reshape(h_1.shape))

    #ploting scatter plot for LDA
    female_LDA = np.log(util.density_Gaussian(mu_female, cov, x))
    male_LDA = np.log(util.density_Gaussian(mu_male, cov, x))

    LDA_decision = np.maximum(female_LDA, male_LDA)
    LDA_index_female = np.equal(female_LDA, LDA_decision)

    LDA_class = np.where(LDA_index_female == True, 2, 1)

    plt.title('lda')
    plt.xlabel('height') #50, 80
    plt.ylabel('weight') #80,280

    plt.xlim([50,80])
    plt.ylim([80, 280])

    male_first = False
    female_first = False

    female_contour = []
    female_width = []
    female_height = []
    male_contour = []

    for i in range(0, x.shape[0]):
        if(LDA_class[i]) == 1 and not male_first:
            plt.scatter(height[i], width[i], s=6, label='male', color='blue')
            male_first = True
            male_contour.append([height[i],width[i]])
        elif (LDA_class[i]) == 2 and not female_first:
            plt.scatter(height[i], width[i], s=6, label='female', color='red')
            female_first = True
            female_contour.append([height[i], width[i]])
            female_width.append(width[i])
            female_height.append(height[i])
        elif (LDA_class[i]) == 1:
            plt.scatter(height[i], width[i], s=6, color='blue')
            male_contour.append([height[i], width[i]])
        elif (LDA_class[i]) == 2:
            plt.scatter(height[i], width[i], s=6, color='red')
            female_contour.append([height[i], width[i]])
            female_width.append(width[i])
            female_height.append(height[i])

    #plt.legend({'female', 'male'})

    plt.show()

    plt.figure(2)

    #decision boundary calculations for QDA
    height_range = np.linspace(50, 80,100)
    width_range = np.linspace(80, 280, 100)
    h_1, w_1 = np.meshgrid(height_range, width_range)
    H_1 = h_1.ravel()
    W_1 = w_1.ravel()

    female_QDA_2 = util.density_Gaussian(mu_female, cov_female, np.column_stack((H_1, W_1)))
    male_QDA_2 = util.density_Gaussian(mu_male, cov_male, np.column_stack((H_1, W_1)))
    function_QDA = (male_QDA_2.T - female_QDA_2.T).reshape(h_1.shape)

    #plotting contours for QDA
    plt.contour(h_1, w_1, function_QDA, levels=[0])
    plt.contour(h_1, w_1, female_QDA_2.reshape(h_1.shape))
    plt.contour(h_1, w_1, male_QDA_2.reshape(h_1.shape))

    #plotting scatter plots for QDA
    female_QDA = np.log(util.density_Gaussian(mu_female, cov_female, x))
    male_QDA = np.log(util.density_Gaussian(mu_male, cov_male, x))

    QDA_decision = np.maximum(female_QDA, male_QDA)
    QDA_index_female = np.equal(female_QDA, QDA_decision)

    QDA_class = np.where(QDA_index_female == True, 2, 1)

    plt.title('qda')
    plt.xlabel('height') #50, 80
    plt.ylabel('weight') #80,280

    plt.xlim([50,80])
    plt.ylim([80, 280])

    male_first = False
    female_first = False

    for i in range(0, x.shape[0]):
        if(QDA_class[i]) == 1 and not male_first:
            plt.scatter(height[i], width[i], s=6, label='male', color='blue')
            male_first = True
        elif (QDA_class[i]) == 2 and not female_first:
            plt.scatter(height[i], width[i], s=6, label='female', color='red')
            female_first = True
        elif (QDA_class[i]) == 1:
            plt.scatter(height[i], width[i], s=6, color='blue')
        elif (QDA_class[i]) == 2:
            plt.scatter(height[i], width[i], s=6, color='red')

    plt.show()

    return (mu_male,mu_female,cov,cov_male,cov_female)
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testi  ng set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """

    label_male = np.count_nonzero(y == 1)
    label_female = np.count_nonzero(y == 2)
    print('number of male, female, total', label_male, label_female, y.shape)

    #LDA classification
    female_LDA = np.log(util.density_Gaussian(mu_female, cov, x))
    male_LDA = np.log(util.density_Gaussian(mu_male, cov, x))

    LDA_decision = np.maximum(female_LDA, male_LDA)
    LDA_index_female = np.equal(female_LDA, LDA_decision)

    LDA_class = np.where(LDA_index_female == True, 2, 1)

    #find all male correct instances
    lda_male = np.where(np.where(LDA_class == 1, 1, 0) == y, True, False)

    #find all female correct instances
    lda_female = np.where(np.where(LDA_class == 2, 2, 0) == y, True, False)

    lda_mis_male = label_male - np.count_nonzero(lda_male == True)
    lda_mis_female = label_female - np.count_nonzero(lda_female == True)

    mis_lda = (lda_mis_male + lda_mis_female)/y.shape[0]*100

    #QDA classification
    female_QDA = np.log(util.density_Gaussian(mu_female, cov_female, x))
    male_QDA = np.log(util.density_Gaussian(mu_male, cov_male, x))

    QDA_decision = np.maximum(female_QDA, male_QDA)
    QDA_index_female = np.equal(female_QDA, QDA_decision)

    QDA_class = np.where(QDA_index_female == True, 2, 1)

    #find all male correct instances
    qda_male = np.where(np.where(QDA_class == 1, 1, 0) == y, True, False)

    #find all female correct instances
    qda_female = np.where(np.where(QDA_class == 2, 2, 0) == y, True, False)

    qda_mis_male = label_male - np.count_nonzero(qda_male == True)
    qda_mis_female = label_female - np.count_nonzero(qda_female == True)

    mis_qda = (qda_mis_male + qda_mis_female)/y.shape[0]*100

    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')

    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)

    print(cov)
    print(cov_female)
    print(cov_male)

    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)
    
    print(mis_LDA, mis_QDA)
