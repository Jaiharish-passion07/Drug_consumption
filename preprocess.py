import numpy as np
import pandas as pd

#ordinal encoding
age_dict={'18-24':6,'25-34':5,'35-44':4,'45-54':3,'55-64':2,'65+':1}
education_dict={'Doct Deg':9,'Mast Deg':8,'Prof Cert':7,'Univ Deg':6,'Some Clg':5,
           'LS@18Y':4,'LS@17Y':3,'LS@16Y':2,'LSB 16Y':1}
ethnicity_list=['Asian','Black','Mixed-Black/Asian','Mixed-White/Asian','Mixed-White/Black','White','other']
gender_list=['Female','Male']
country_list=['Australia', 'Canada', 'New Zealand','Other', 'Reb of Inreland', 'UK', 'USA']

def encdoding_data(age,education,n_score_values,e_score_values,o_score_values,a_score_values,c_score_values,impulsive_val,ss_val,country,ethinicty,gender):
    features = [age_dict[age],education_dict[education],n_score_values,e_score_values,
                o_score_values,a_score_values,c_score_values,impulsive_val,ss_val]
    encode_1 = pd.DataFrame(data={'ethni': ethnicity_list})
    encode_2 = pd.DataFrame(data={'Gender': gender_list})
    encode_3 = pd.DataFrame(data={'country': country_list})

    cou_enc = pd.get_dummies(encode_3['country']).iloc[country_list.index(country), :].values
    eth_enc = pd.get_dummies(encode_1['ethni']).iloc[ethnicity_list.index(ethinicty), :].values
    gen_enc = pd.get_dummies(encode_2['Gender']).iloc[gender_list.index(gender), :].values

    a, b,c = list(cou_enc), list(eth_enc),list(gen_enc)

    features.extend(a)
    features.extend(b)
    features.extend(c)
    in_arr=np.array([features],dtype=float)

    return in_arr