import streamlit as st
import pandas as pd
import numpy as np
import requests
import shap
import json
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

TIMEOUT = (5, 30)
MAIN_COLUMNS = ['CNT_CHILDREN','APPS_EXT_SOURCE_MEAN', 'APPS_GOODS_CREDIT_RATIO',
                'AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED']

API_URL = "https://kiliandatadev.pythonanywhere.com/"

def st_shap(plot, height=None):
    """ Create a shap html component """
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def get_columns_mean():
    """ Get customers main columns mean values """
    response = requests.get(API_URL + "columns/mean", timeout=TIMEOUT)
    content = response.json()
    return pd.Series(content)
 
def get_columns_neighbors(cust_id):
    """ Get customers neighbors main columns mean values """
    response = requests.get(API_URL + "columns/neighbors/id/" + str(cust_id))
    content = response.json()
    return pd.Series(content)

def make_prediction(row_data):
	""" Get prediction of the model and display infos """
	response = requests.post(API_URL + "prediction", json=row_data)
	if response.status_code == 200:
		prediction = response.json()["body"][0]
		proba_authorized = prediction[0]
		result = "Crédit autorisé" if proba_authorized > 0.5 else "Crédit refusé"
		st.text("Les crédits sont autorisés si votre score client est supérieur à 50 points")
		st.text("Score : " + str(round(proba_authorized * 100, 1)) + "/100 points")
		st.text(result)
	else:
		st.write('Erreur lors de la requête à l\'API')

def get_shap_values(row_data):
	""" Get customer's shap values and display pie chart of top3 most important features"""
	response = requests.post(API_URL + "shap-values", json=row_data)
	if response.status_code == 200:
		result = response.json()["body"]
		shap_values_cleaned = np.array(result["shap_values"])
		
		df_feature_importance = pd.DataFrame(data=shap_values_cleaned[0], index=[id_selected], columns=data.columns)
		df_feature_importance_abs = df_feature_importance.abs()
		df_user_feat_imp = df_feature_importance_abs.iloc[0, :]
		top3_features = df_user_feat_imp.sort_values(ascending=False).head(3)
		
		st.text("Ces 3 variables participent le plus à la décision d'attribution du crédit")
		st.text("Voici leur répartition en fonction de leur importance :")
		fig = plt.Figure(figsize=(4, 4))
		ax = fig.subplots()
		ax.pie(top3_features, labels = top3_features.index, autopct='%.0f%%', colors=['lightgreen', 'skyblue', 'orange'], shadow=True)
		st.pyplot(fig)
	else:
		st.write('Erreur lors de la requête à l\'API')

st.title('Prêt à dépenser : Scoring Crédit')
st.image('./assets/images/logo-app.png', width=250)

st.subheader('Tableau de données de nos clients')
data = pd.read_csv('./data/X_train_sample.csv', index_col=[0])
st.dataframe(data)

st.subheader("Sélectionner un client")
ids_index = data.index

id_selected = st.selectbox('Sélectionner un ID', options=ids_index, index=0)
iloc_id_selected = ids_index.get_loc(id_selected)
colonnes_selectionnees = st.multiselect('Filtrer par colonne', data.columns)
df_filtre = data.loc[id_selected, colonnes_selectionnees] if colonnes_selectionnees else data.loc[id_selected]

if id_selected:
	st.write('ID client sélectionné :', id_selected)
	row_data = data.loc[id_selected].to_dict()
	st.dataframe(df_filtre)

st.subheader("Lancer une simulation de crédit")
go_button = st.button('🤖')
if go_button :
	with st.spinner(text='Chargement'):
		st.subheader("Resultats")
		make_prediction(row_data)

		st.subheader("En savoir plus sur ce résulat")
		get_shap_values(row_data)
		using_shap_plots = False
		if using_shap_plots:
				# shap_values_full = json.loads(result["shap_values_full"])
				# shap_values = shap_values_full["values"]
				# st.write(np.shape(expected_value))
				# st.write(np.shape(shap_values_cleaned))
				# st.write('shape:', np.array(shap_values_cleaned).shape)
				# st.write(np.shape(shap_values[0][0]))
				# st.write(np.shape(data))
				# st.write(shap_values)
				# st.write(shap_values_cleaned)
				st.write('data:', data.iloc[0].shape)
				# st.write('shap:', shap_values_cleaned)
				# st.write('expec:', expected_value)

				# shap_object = ShapObject(base_values = expected_value[1],
				#          values = shap_values_cleaned[1][0,:],
				#          feature_names = data.columns,
				#          data = data.iloc[0,:],
				# 		 display_data = data.iloc[0,:])

				# shap_input = ShapObject(expected_value[0], shap_values_cleaned[0], 
				#        data.iloc[0,:], feature_names=data.columns, display_data=data.iloc[0,:])

				# shap.waterfall_plot(shap_input)

				# shap.plots.waterfall(shap_object)
				
				# shap.waterfall_plot(shap_object)

				# st_shap(shap.plots.decision(expected_value[1], shap_values_cleaned[1], row_scaled, link="logit"))

				# shap_explanation = shap.Explanation(shap_values, base_values=expected_value, data=shap_values_full["data"])
				# _____________________________________________________________________________________________
				# Waterfall plots
				# st_shap(shap.plots._waterfall.waterfall_legacy(expected_value[0], np.array(shap_values_cleaned)))
				# IndexError: index 212 is out of bounds for axis 0 with size 2

				# st_shap(shap.plots.waterfall(expected_value, shap_values))
				# base_values = shap_values.base_values
				# AttributeError: 'list' object has no attribute 'base_values'

				# st_shap(shap.waterfall_plot(expected_value, shap_values))
				# AttributeError: 'list' object has no attribute 'base_values'

				# st_shap(shap.waterfall(shap_values))
				# AttributeError: module 'shap' has no attribute 'waterfall'
				# _____________________________________________________________________________________________
				# Other plots
				# st_shap(shap.force_plot(expected_value[0], shap_values_cleaned[0][0], data.iloc[0,:]))
				# AssertionError: The shap_values arg looks multi output, try shap_values[i] avec shap_values_cleaned[0][0]
				# AssertionError: visualize() can only display Explanation objects (or arrays of them) avec shap_values_cleaned[0][0][0]

				# st_shap(shap.force_plot(shap_explanation))
				# Exception: In v0.20 force_plot now requires the base value as the first parameter! Try shap.force_plot(explainer.expected_value, shap_values) 
				# or for multi-output models try shap.force_plot(explainer.expected_value[0], shap_values[0]).

				# shap.plots.decision(expected_value[0], shap_values_cleaned[0], data.iloc[1,:], link="logit")
				# TypeError: Looks like multi output. Try base_value[i] and shap_values[i], or use shap.multioutput_decision_plot().

				# Renvoient None
				# st_shap(shap.multioutput_decision_plot(expected_value, shap_values_cleaned, row_index=0))

				# shap.summary_plot(np.array(shap_values_cleaned), data.iloc[0,:])
				# IndexError: index 212 is out of bounds for axis 1 with size 1

				# shap.plots.bar(shap_values)

		st.subheader("Comparer avec des clients similaires")
		customer_df = data[MAIN_COLUMNS].loc[id_selected]
		neighbors_df = get_columns_neighbors(iloc_id_selected).rename("Moyennes des clients similaires")
		mean_df = get_columns_mean().rename("Moyennes de tous les clients")
		concat_df = pd.concat([customer_df, neighbors_df, mean_df], axis=1)
		st.dataframe(concat_df)
		for col in MAIN_COLUMNS:
			df_for_chart = concat_df.loc[[col]] 
			df_for_chart = df_for_chart.transpose()
			df_for_chart.columns = [col]
			st.bar_chart(df_for_chart)
