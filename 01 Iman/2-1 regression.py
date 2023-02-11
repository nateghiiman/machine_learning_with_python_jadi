import pandas as pd
from matplotlib import pyplot as plt
from collections import OrderedDict
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def df_clean(df):
    ### Some Area datas were not correct so we remove the corresponding rows
    df=df[df.Area.apply(lambda x: x.isnumeric())]
    ### Change data type of column "Area"
    #df["Area"]=pd.to_numeric(df.Area)
    df = df.astype({"Area": float})  #this is another way!
    ### Transform Boolean to int
    df=df.astype({"Elevator":int,"Warehouse":int,"Parking":int})
    ### Some house's Areas were too big and we were asked to ignore them
    df=df[df.Area.apply(lambda x: x<400)]
    ### Some house's Addresses are NaN and we must ignore them
    df=df[df.Address.apply(lambda x: not(pd.isnull(x)))]
    ### Transform Address to numerical values
    df["PricePerMeter"]=df['Price']/df["Area"]
    df=df.sort_values(by=['PricePerMeter'])
    Address_list=OrderedDict.fromkeys(df.Address,0)
    counter=1
    for i in Address_list:
        Address_list[i]=counter
        counter+=1
    df["AddressN"]=df.Address.apply(lambda x : Address_list[x])
    ### column names so far are:
    ###['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address', 'Price', 'Price(USD)', 'PricePerMeter', 'AddressN']
    return(df)



### choose data frame






### assess if there is a linear correlation using scatterplot
def single_reg(cdf,y):
    
    ### create train and test data sets
    mask=np.random.rand(len(cdf))<0.8
    train=cdf[mask]
    test=cdf[~mask]
    figscatters=plt.figure()
    counterscatter=0
    for i in cdf.columns.values:
        if i==y:
            pass
        else:
            counterscatter+=1
            print("##########>Single regression: X=%s and Y=%s<##########" % (i,y))
            ax=figscatters.add_subplot(int("32"+str(counterscatter)))

            x_train=np.asanyarray(train[[i]])
            y_train=np.asanyarray(train[[y]])
            #x_train=train[[i]]
            #y_train=train[[y]]
            x_test=np.asanyarray(test[[i]])
            y_test=np.asanyarray(test[[y]])
            #x_test=test[[x]]
            #y_test=test[[y]]

            ax.scatter(x_train,y_train)
            plt.xlabel(i)
            plt.ylabel(y)

            reg=LinearRegression()
            reg.fit(x_train,y_train)
            #print("coef=",reg.coef_[0][0])
            #print("intercept=",reg.intercept_[0])
            ax.plot(x_train,reg.intercept_+reg.coef_[0][0]*x_train,'-r')

            y_predict=reg.predict(x_test)
            print("R2=",r2_score(y_predict,y_test))
    plt.show()
    


def multiple_reg(cdf,select,y):
    mask=np.random.rand(len(cdf))<0.8
    train=cdf[mask]
    test=cdf[~mask]
    reg=LinearRegression()
    x_train=np.asanyarray(train[select])
    y_train=np.asanyarray(train[[y]])
    x_test=np.asanyarray(test[select])
    y_test=np.asanyarray(test[[y]])
    reg.fit(x_train,y_train)
    print("coef=",reg.coef_[0])
    print("intercept=",reg.intercept_[0])
    y_predict=reg.predict(x_test)
    score=reg.score(x_test,y_test)
    RSS=np.mean((y_predict-y_test)**2)
    print("Residual Sum of squares= %f" %RSS)
    print("Variance score= %F" % score)
    print("R2 score=%f" % r2_score(y_predict,y_test))


df=pd.read_csv("C:/python/Jadi_ML/01 Iman/2-1 HousePriceData.csv",index_col=False)
df=df_clean(df)
cdf=df[["Area",'Room', 'Parking', 'Warehouse', 'Elevator','Price(USD)','AddressN']]
### assess data with hostogram
#cdf.hist()
#plt.show()   
single_reg(cdf,"Price(USD)")
###you can select variables as x of regression:
select=["Area",'AddressN']
multiple_reg(cdf,select,"Price(USD)")