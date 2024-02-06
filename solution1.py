import csv
import math
import numpy
from csv import DictReader
from scipy.stats import lognorm, beta
from matplotlib import pyplot as plt

# Creating a function to validate csv file   
def validate_csvfile():
    with open("SCC.444 Risk Modelling Expert Data Capture.csv", 'r') as file:
        # Reading csv file using DictReader
        creader = csv.DictReader(file)
        crows=[]
        cerrors=[]
        alphanum=[]
        count=1
        # Iterating through each row to validare csv data
        for row in creader:
            errmsg=[]
            for key,value in row.items():
                # Checking the values in csv file, if it is numeric and converting it into int or flaot data types
                if(value.isnumeric()):
                    if (int(value)):
                        row[key]= int(value)
                    else:
                        row[key]=float(value)
                else:
                    if(key!='Title'):
                        alphanum.append(key)
                # checking if the values in csv file are null or empty
                if(value== 'null' or value=='' or value==' '):
                    errmsg.append("Missing values ")
                    
            # Checking if prob_min, prob_most, prob_max, lb_loss and ub_loss are of int or float data types
            # and prints Invalid datatypes error message      
            if not ((type(row['prob_min']==int) or  type(row['prob_min']==float))
            and (type(row['prob_most']==int) or  type(row['prob_most']==float))
            and (type(row['prob_max']==int) or  type(row['prob_max']==float))
            and (type(row['lb_loss']==int) or  type(row['lb_loss']==float))
            and (type(row['ub_loss']==int) or  type(row['ub_loss']==float))):
                errmsg.append("Invalid datatypes")
                
            # checking the range of prob_min, prob_most and prob_max as per given conditions 
            # prints error message if the condition does not satisfy 
            if('prob_min' not in alphanum and 'prob_most' not in alphanum and 'prob_max' not in alphanum):
                if not (0 <= row['prob_min'] <= row['prob_most'] <= row['prob_max']):
                    errmsg.append("Invalid range prob_min, prob_most and prob_max")
                    
            # checking the range of lb_loss and ub_loss as per given conditions 
            # prints error message if the condition does not satisfy        
            if('lb_loss' not in alphanum and 'ub_loss' not in alphanum ):
                if not (0 <= row['lb_loss'] < row['ub_loss']):
                    errmsg.append("Invalid range lb_loss and ub_loss")
                    
            # If no error message in the csv, appending that row to crows
            if(len(errmsg)==0):
                row['id']=count
                crows.append(row)
            # If there is an error message in the csv, appending that row to cerrors    
            else:
                row['errmsg']= errmsg
                row['id']=count
                cerrors.append(row)
            # Using count to store and display the line numbers in csv file    
            count+=1
        # Returning csv file rows and csv file errors    
        return crows,cerrors
    
# Creating a function for Distribution modelling using PERT and LogNormal Distribution    
def distribution_modelling(crows):
    for item in crows:
        # calcualting alpha value 'a' using prob_min, prob_max and prob_most for PERT distribution
        a= 1+4*((item['prob_most']-item['prob_min'])/(item['prob_max']-item['prob_min']))
        # calcualting beta value 'b' using prob_min, prob_max and prob_most for PERT distribution
        b= 1+4*((item['prob_max']-item['prob_most'])/(item['prob_max']-item['prob_min']))
        # Generating the random value for PERT distribution using alpha 'a' and beta 'b'
        pvalue= beta.rvs(a, b, loc=item['prob_most'], scale=item['prob_max']-item['prob_min'])
        pval.append(pvalue)
        # calculating pert mean by using prob_min, prob_most and prob_max
        item['pertmean']= (item['prob_min']+4*item['prob_most']+item['prob_max'])/6
        # calculating pert median by using prob_min, prob_most and prob_max
        item['pertmedian']= (item['prob_min']+6*item['prob_most']+item['prob_max'])/8
        # calculating pert mode by using prob_most 
        item['pertmode']=item['prob_most']
        # calculating pert standard deviation by using pertmean, prob_min and prob_max
        item['pertstdev']=math.sqrt(((item['pertmean']-item['prob_min'])*(item['prob_max']-item['pertmean']))/7)
        # calculating pert Annualised Loss Expectancy by using pertmean and prob_most
        item['pertale']= item['prob_most']*item['pertmean']
        pertdist.append(item)
        # calculating lognormal distribution mu using ub_loss and lb_loss
        item['lognmu']=((math.log(item['ub_loss'])) + (math.log(item['lb_loss'])))/2
        # calculating lognormal distribution sigma using ub_loss and lb_loss
        item['lognsigma']=((math.log(item['ub_loss'])) - (math.log(item['lb_loss'])))/3.29
        # Generating random values for Lognormal Distribution using mu and sigma
        lognvalue=lognorm(s=item['lognsigma'], scale=math.exp(item['lognmu']))
        logval.append(lognvalue)
        # calculating lognormal distribution mean using mu and sigma
        item['lognmean']= math.e ** (item['lognmu']+ ((item['lognsigma']**2)/2))
        # calculating lognormal distribution median using mu 
        item['lognmedian']= math.e **(item['lognmu'])
        # calculating lognormal distribution mode using mu and sigma
        item['lognmode']=math.e **(item['lognmu']-(item['lognsigma']**2))
        # calculating lognormal distribution variance using sigma and mu
        lognvariance = (math.exp(item['lognsigma']*2) - 1)*math.exp(2*item['lognmu'] + item['lognsigma']*2)
        # calculating lognormal distribution standard deviation using lognormal variance
        item['lognstdev'] = math.sqrt(lognvariance)
        # calculating lognormal distribution Annualised Loss Expectancy using prob_most and lognormal mean
        item['lognale']= item['prob_most']*item['lognmean']
        logndist.append(item)
        plog_dist.append(item)
        # Displaying the PERT and Log Normal Distribution parameters
        print(f"PERT Distribution parameters :")
        print(f"Mean:{item['pertmean']}")
        print(f"Median:{item['pertmedian']}")
        print(f"Mode:{item['pertmode']}")
        print(f"Standard deviation:{item['pertstdev']}")
        print(f"Annual Loss Expectancy:{item['pertale']}")
        print(f"Lognorm Distribution parameters :")
        print(f"Mean:{item['lognmean']}")
        print(f"Median:{item['lognmedian']}")
        print(f"Mode:{item['lognmode']}")
        print(f"Standard deviation:{item['lognstdev']}")
        print(f"Annual Loss Expectancy:{item['lognale']}")
        print(f"mu:{item['lognmu']}")
        print(f"sigma:{item['logsigma']}")
        # Returning Pert log distribution
    return plog_dist

# Creating a function for Monte Carlo simulation using Distribution Modelling    
def monte_carlo_simulation(revent,rlosses,plog_dist):
    # Looping to calculate loss and events occurred till 5000 years 
    for year in range(5000):
        lsum=[]
        levent=[]
        # looping through each risk to calculate loss and event occurred
        for risk in plog_dist :
            loss=0
            # calcualting alpha value 'a' using prob_min, prob_max and prob_most for Monte Carlo Simulation
            a= 1+4*((risk['prob_most']-risk['prob_min'])/(risk['prob_max']-risk['prob_min']))
            # calcualting beta value 'b' using prob_min, prob_max and prob_most for Monte Carlo Simulation
            b= 1+4*((risk['prob_max']-risk['prob_most'])/(risk['prob_max']-risk['prob_min']))
            # Generating the random value for Monte Carlo Simulation using alpha 'a', beta 'b', prob_max and prob_most
            rand_pert=beta.rvs(a,b,loc=risk['prob_most'],scale=risk['prob_max']-risk['prob_most'])
            # Calculating the sum of loss by iterating through each instance
            for pert in range(1, int(rand_pert)+1):
              loss +=lognorm.rvs(s=(risk['lognsigma']), loc=risk['lb_loss'],scale=risk['ub_loss']-risk['lb_loss'])
            lsum.append(loss)
            levent.append(rand_pert)
            # Assigning the calculated values(loss and event occurred) to each risk
            if(len(revent[risk['Title']])>0 and len(rlosses[risk['Title']])>0):
              revent[risk['Title']] +=[rand_pert]
              rlosses[risk['Title']] +=[loss]
            else:
              revent[risk['Title']] =[rand_pert]
              rlosses[risk['Title']] =[loss]
        # printing Average, Min and Max no of events occurred per year      
        print(f"Average no of events per year {numpy.average(levent)}")
        print(f"Min no of events per year  {numpy.min(levent)}")
        print(f"Max no of events per year  {numpy.max(levent)}")
        # printing Average, Min and Max losses occurred in a year 
        print(f"Average Loss in a year {numpy.average(lsum)}")
        print(f"Min loss in a year {numpy.min(lsum)}")
        print(f"Max loss in a year {numpy.max(lsum)}")
        
    # calculating Loss Exceedance value for 75%, 50% and 25%
    losses = numpy.array([numpy.percentile((lsum), x) for x in range(1, 100, 1)])
    frequency = numpy.array([float(100 - x) / 100.0 for x in range(1, 100, 1)])
    print(f"There is a {frequency[75] * 100}% of losses exceeding {losses[75]:.2f} or more")
    print(f"There is a {frequency[50] * 100}% of losses exceeding {losses[50]:.2f} or more")
    print(f"There is a {frequency[25] * 100}% of losses exceeding {losses[25]:.2f} or more")
    # Returning events and losses occurred for each risk, loss exceedance values and sum of losses
    return revent,rlosses,losses,frequency,lsum

# Creating a function to diplay events and losses occurred for each risk   
def revent_loss(revent,rlosses):
    for key,value in revent.items():
        print(key)
        print(f"Total no of times risk event occurred {numpy.sum(value)}")
        print(f"Mean Loss {numpy.average(rlosses[key])}")
        print(f"Min loss {numpy.min(rlosses[key])}")
        print(f"Max loss {numpy.max(rlosses[key])}")

# Creating a data visualisation function which represent the data from the analysis of the risks   
def data_visualisation(losses,percentiles,lsum):
    plt.subplot(1,2,1)
    # Displays loss exceedance curve of all losses in a year
    plt.plot(losses, percentiles)
    # displays the graph title as 'Aggregated Loss Exceedance'
    title="Aggregated Loss Exceedance"
    plt.title(title)
    # Labelling x-axis as "losses"
    plt.xlabel ('losses')
    # Labelling y-axis as "percentiles"
    plt.ylabel ('percentiles')
    
    plt.subplot(1,2,2)
    # Displays scatter plot with title 'Loss scatter visualisation' to visualise the data
    plt.title("Loss scatter visualisation")
    # calculating x and y axes values to display sactter plot
    x= numpy.linspace(0,len(lsum),len(lsum))
    y= x+ 10 * numpy.random.randn(len (lsum))
    # Labelling x-axis as "losses"
    plt.xlabel('losses')
    # Labelling y-axis as "probability"
    plt.ylabel('probability')
    plt.scatter(x,y)
    plt.show()

# Main code to check the functionality             
if __name__=="__main__":
    # Calling validation function to validate errors in csv file
    crows,cerrors= validate_csvfile()
    for item in cerrors:
        print (f"Id: {item['id']}, Errors: {item['errmsg']}, values: {item['Title']}, {item['prob_min']},{item['prob_most']},{item['prob_max']},{item['lb_loss']},{item['ub_loss']} ")
    # Initialising list of values    
    pertdist=[]
    pval=[]
    logndist=[]
    logval=[]
    plog_dist=[]
    a=0
    b=0
    # Calling distribution_modelling function to get pert log distribution
    pert_log=distribution_modelling(crows)
    revent={}
    rlosses={}
    for val in crows:
        revent[val['Title']]= []
        rlosses[val['Title']] =[]
    # Calling monte_carlo_simulation function to calculate events and losses occurred for each risk, sum of losses, loss and percentile of risks    
    revent,risk_loss,loss,percentile,lsum=monte_carlo_simulation(revent,rlosses,pert_log)
    # Calling revent_loss function to calculate total no.of times the risk events occurred, mean, min_loss and max_loss
    revent_loss(revent,risk_loss)
    # Calling data_visualisation function to visualise the data
    data_visualisation(loss,percentile,lsum)