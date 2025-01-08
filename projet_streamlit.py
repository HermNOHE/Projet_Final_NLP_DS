import streamlit as st
from lime import lime_text
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import joblib
from sklearn.model_selection import train_test_split
import plotly.figure_factory as ff
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

df = pd.read_csv('reviews_trust.csv')

x_data = pd.read_csv('x_data.csv')
X_test = joblib.load('X_test.pkl')
y_test = joblib.load('y_test.pkl')
leaderboard = pd.read_csv("automl_leaderboard.csv", sep = ',')
loaded_model = joblib.load('gbm.pkl')
loaded_tfidf = joblib.load('tfidf_vectorizer.pkl')

st.sidebar.title("Sommaire")
choix_graphique = ["Tous", "Répartion des commentaires par source","Distribution des notes selon la source",
                   "Répartition des réponses", "Pourcentage des notes des commentaires dans les cas de non réponses", "Evolution des notes"]

pages = ["Contexte du projet", "Analyse des données", "Modélisation"]
page = st.sidebar.radio("Aller vers la page :", pages)

columns_list = df.columns
descrip = ["Commentaire du client", "La note donnée par le client", "La date du commentaire", "Nom du client", "La réponse donnée par la socièté", 
           "L'origine des données (TrustPilot,TrustedShop)", "La socièté (TrustPilot,TrustedShop)", "La ville du client", "la date de mise à jour", "La date de la commande",
           "Retard sur la réponse attendue"]

col_descr = pd.DataFrame({
    "Feature": columns_list,
    "Description": descrip
})
rows, cols = df.shape

### visuel valeurs manquantes

val_missing = pd.DataFrame({
    'Colonnes': df.columns,
    'Valeurs': round(100 * df.isna().sum() / len(df),4)
}).reset_index(drop=True)

val_missing = val_missing.sort_values(by='Valeurs', ascending=True)

#### Dashboard

reponse_statut = df['reponse'].notna().value_counts(normalize=True) * 100

labels = ["Réponse donnée", "Pas de réponse"]
values = reponse_statut.values
colors = ["green", "orange"]

commentaires_without_reply = df[df['reponse'].isna() == True]
commentaires_without_reply = commentaires_without_reply.groupby(['source', 'star']).agg(nbr = ('Commentaire', 'count')).reset_index()
totals = commentaires_without_reply.groupby('source')['nbr'].transform('sum')
commentaires_without_reply['percentage'] = (commentaires_without_reply['nbr'] / totals) * 100

## Evolution des notes dans le temps

df_analyse = df.dropna(subset= ['Commentaire', 'date'], how = 'all', axis = 0)
df_analyse = df_analyse[['Commentaire', 'star', 'date', 'reponse']]
df_analyse['date'] = df_analyse['date'].astype('str')
df_analyse['date'] = df_analyse['date'].apply(lambda x: x[:10])
df_analyse['date'] = pd.to_datetime(df_analyse['date'], format='mixed', errors='coerce')
df_analyse = df_analyse.dropna(subset= ['date'], how = 'all', axis = 0)

df_analyse['mois'] = df_analyse['date'].dt.month
df_analyse['trimestre'] = df_analyse['date'].dt.quarter
df_analyse['annee'] = df_analyse['date'].dt.year      
df_analyse['mois'] = df_analyse['mois'].astype(int)
df_analyse['annee'] = df_analyse['annee'].astype(int)

df_monthly = df_analyse.groupby(['annee', 'mois', 'star'])['Commentaire'].count().reset_index()
df_quarterly = df_analyse.groupby(['annee', 'trimestre', 'star'])['Commentaire'].count().reset_index()
df_monthly['mois_annee'] = df_monthly['mois'].astype(str) + '-' + df_monthly['annee'].astype(str)
df_monthly['mois_annee'] = pd.to_datetime(df_monthly['mois_annee'], format='%m-%Y')


fig_na= px.bar(
    val_missing,
    x='Colonnes',
    y='Valeurs',
    text='Valeurs',
    color='Valeurs',
    color_continuous_scale='viridis',
    title="Pourcentage de valeurs manquantes par variable"
)

fig_na.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
fig_na.update_layout(
    yaxis_title="Valeurs Manquantes (%)",
    xaxis_title="Variables",
    xaxis=dict(tickangle=45),  # Rotation des étiquettes
    margin=dict(t=50, b=50, l=25, r=25)
)

fig = make_subplots(
    rows=3, cols=2, 
    specs=[
        [{"type": "xy"}, {"type": "xy"}], [{"type": "domain"}, {"type": "xy"}],[{"type": "xy"}, None]])

# 1er graphique
fig1 = px.histogram(data_frame=df, x="source", barmode="group", 
                    color_discrete_sequence=px.colors.sequential.Viridis[:2])
for trace in fig1.data:
    fig.add_trace(trace, row=1, col=1)
fig1.update_layout(title_text=choix_graphique[1])

# 2e graphique
fig2 = px.histogram(data_frame=df, x="star", color="source", barmode="group", 
                    color_discrete_sequence=px.colors.sequential.Bluered) #Bluered
for trace in fig2.data:
    fig.add_trace(trace, row=1, col=2)
fig2.update_layout(title_text=choix_graphique[2])

# 3e graphique (Pie Chart)
fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, marker=dict(colors=colors), textinfo='percent+label')])
for trace in fig3.data:
    fig.add_trace(trace, row=2, col=1)
fig3.update_layout(title_text=choix_graphique[3])

# 4e graphique
fig4 = px.bar(
    commentaires_without_reply,
    x='source', y='percentage', color='star',
    labels={'source': 'Source', 'percentage': 'Pourcentage', 'star': 'Étoiles'}
)
fig4.update_layout(title_text=choix_graphique[4])
for trace in fig4.data:
    fig.add_trace(trace, row=2, col=2)

# 5e graphique
fig5 = px.line(
    data_frame=df_monthly, x='mois_annee', y='Commentaire', color='star', markers=True,
    title="Évolution des notes par mois",
    labels={"mois_annee": "Mois", "Commentaire": "Effectif", "star": "Note"}
)
for trace in fig5.data:
    fig.add_trace(trace, row=3, col=1)
fig5.update_layout(title_text=choix_graphique[5], legend_title="Note")

# Mise à jour de la mise en page
fig.update_layout(title_text="Analyse graphique des données", height=900, width=1200)

if page == pages[0]:
    
    st.subheader("Contexte du projet")

    st.write("Ce projet s'inscrit dans un contexte d'analyse des commentaires et des notes attribuées par les clients suite à un achat dans un articles sur le site commercial.")

    st.write("Nous disposons d'un jeu de données d'une taille de ")
    st.write("###### Taille du DataFrame :")
    st.write(f"Le jeu de données contient **{rows} lignes** et **{cols} colonnes**.")

    st.write("##### Liste des variables")
    st.table(col_descr)
    st.image("site_comm.jpg", use_container_width =True)

elif page == pages[1]:
    st.write("### Analyse Exploratoire")

    st.write("##### Analyse des valeurs manquantes")
    st.plotly_chart(fig_na)

    #st.write("Analyse combinée")
    #st.plotly_chart(fig1, use_container_width=True)

    graphe = st.selectbox(label = "Graphique", options = choix_graphique)

    if graphe == choix_graphique[0]:
        st.plotly_chart(fig, use_container_width=True)
    elif graphe == choix_graphique[1]:
        st.plotly_chart(fig1, use_container_width=True)
    elif graphe == choix_graphique[2]:
        st.plotly_chart(fig2, use_container_width=True)
    elif graphe == choix_graphique[3]:
        st.plotly_chart(fig3, use_container_width=True)
    elif graphe == choix_graphique[4]:
        st.plotly_chart(fig4, use_container_width=True)
    elif graphe == choix_graphique[5]:
        st.plotly_chart(fig5, use_container_width=True)

elif page == pages[2]:
    st.write("#### utilisation du Gradient Boosting Classifier")

    y_pred = loaded_model.predict(X_test)
    conf_matrix_vf = pd.crosstab(y_test, y_pred, rownames=['Classe réelle'], colnames=['Classe prédite'])

    report = classification_report(y_test, y_pred, output_dict=True)

    # Convertir en DataFrame pour un affichage plus clair
    report_df = pd.DataFrame(report).transpose().reset_index()
    report_df.rename(columns={"index": "Classe"}, inplace=True)

    z = conf_matrix_vf.values
    x_labels = conf_matrix_vf.columns.astype(str)
    y_labels = conf_matrix_vf.index.astype(str)

    fig = ff.create_annotated_heatmap(
    z, x=list(x_labels), y=list(y_labels), 
    colorscale="greens", showscale=True,
    hoverinfo="z")

    fig.update_layout(title_text="Matrice de Confusion",
    title_x=0.5,xaxis=dict(title="Classe Prédite"),
    yaxis=dict(title="Classe Réelle", autorange="reversed"),  # Inverser l'ordre des labels
    font=dict(size=12))

    #st.write("###### Matrice de Confusion")
    st.plotly_chart(fig)

    ## Classification report

    formatted_report_df = report_df.copy()
    columns_to_format = [col for col in report_df.columns if col != "support"]

    for col in columns_to_format:
        if col != "Classe":  # Ne pas formater la colonne des classes
            formatted_report_df[col] = formatted_report_df[col].apply(
                lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
    
    row_colors = ["lightgreen" if "accuracy" in str(row).lower() else "lightcoral" 
    for row in formatted_report_df["Classe"]]

    # Création du tableau Plotly
    fig_rp = go.Figure(
        data=[go.Table(
                header=dict(
                    values=list(formatted_report_df.columns),
                    fill_color="lightsteelblue",
                    align="center",
                    font=dict(color="black", size=14),
                ),cells=dict(
                    values=[formatted_report_df[col] for col in formatted_report_df.columns],
                     fill_color=[[row_colors[i] for i in range(len(row_colors))]
                    for _ in formatted_report_df.columns],
                    align="center",
                    font=dict(color="black", size=12),),)])
    # Mettre à jour le layout pour l'esthétique
    fig_rp.update_layout(
        title_text="Rapport de Classification",
        title_x=0.5,
        margin=dict(l=0, r=0, t=30, b=0),
        height=200,)

    # Afficher sur Streamlit
    st.plotly_chart(fig_rp)

    st.write("### Affichage de la tendance du modèle avec LIME")

    pipeline = make_pipeline(loaded_tfidf, loaded_model)

    explainer = lime_text.LimeTextExplainer(class_names=loaded_model.classes_)

    idx = st.selectbox("Choisissez un index de texte à expliquer :", np.random.choice(range(1, 51), size=5, replace=False))
    x_data = x_data.astype(str)
    selected_text = list(x_data.iloc[idx])[1]

    exp = explainer.explain_instance(str(selected_text), pipeline.predict_proba, num_features=10)

    st.text_area(f"Texte sélectionné (indiv. n° {idx}):", selected_text, height=100)
    st.write("**Explication Lime :**")
    html_content = exp.as_html()
    html_content = html_content.replace(
    "<body>", "<body style='background-color: white;'>")
    st.components.v1.html(html_content, height=600)

    st.write("#### Recherche du meilleur modèle avec AUTOML")
    metrics_to_plot = ['auc', 'logloss', 'aucpr', 'rmse']
    metric = st.radio("Choisissez la valeur de comparaison :", metrics_to_plot)

    # Génération du graphique avec Plotly
    fig = px.bar(
        leaderboard,
        x='model_id',
        y=metric,
        title=f"Comparaison des modèles selon {metric.upper()}",
        labels={'model_id': 'Modèles', metric: metric.upper()},
        color='model_id'
    )

    # Affichage du graphique
    st.plotly_chart(fig)

    st.write("#### Le rapport de classification du meilleur modèle :")

    data = {
    "Class": ["negatif", "positif", "accuracy", "macro avg", "weighted avg"],
    "precision": [0.72, 0.74, None, 0.73, 0.73],
    "recall": [0.68, 0.78, None, 0.73, 0.73],
    "f1-score": [0.70, 0.76, 0.73, 0.73, 0.73],
    "support": [1568, 1879, 3447, 3447, 3447]}

    report_df = pd.DataFrame(data)
    report_df["precision"] = report_df["precision"].fillna("-")
    report_df["recall"] = report_df["recall"].fillna("-")

    # 3. Afficher avec Plotly Table
    fig = go.Figure(data=[go.Table(header=dict(
                    values=["Class", "Precision", "Recall", "F1-Score", "Support"],
                    fill_color="lightskyblue",align="center",font=dict(color="black", size=14),),
                cells=dict(values=[report_df[col] for col in report_df.columns],
                    fill_color=[["lightpink" if cls == "accuracy" else "lightcyan" for cls in report_df["Class"]],],
                    align="center",font=dict(color="black", size=12),),)])
    fig.update_layout(title="Classification Report")
    st.plotly_chart(fig)

#st.dataframe(df.head())

