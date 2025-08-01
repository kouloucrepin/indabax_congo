import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🌧️ Prédiction des Précipitations",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un design moderne
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    }
    .stMetric > div {
        color: black !important;
    }
    .stMetric label {
        color: black !important;
    }
    .stMetric [data-testid="metric-container"] {
        color: black !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: black !important;
    }
    .stMetric [data-testid="metric-container"] label {
        color: black !important;
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: black;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    h3 {
        color: #2980b9;
    }
    .highlight-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Fonctions du notebook
@st.cache_data
def generate_weather_features(df):
    """Génère les caractéristiques météorologiques"""
    result = df.copy()
    
    # Extraire le jour, mois et jour de la semaine de la colonne DATE
    result['DATE'] = pd.to_datetime(result['DATE'])
    result['day'] = result['DATE'].dt.day
    result['month'] = result['DATE'].dt.month
    result['weekday'] = result['DATE'].dt.weekday

    # Variables dérivées
    result['HUMIDITY_SATURATION'] = result['QV2M'] / (result['RH2M']/100 + 1e-6)
    result['TEMP_PRESSURE_RATIO'] = result['T2M'] / result['PS']
    result['TEMP_HUMIDITY_INDEX'] = result['T2M'] * result['RH2M'] / 100
    result['DEW_POINT_SPREAD'] = result['T2M'] - result['T2MDEW']
    result['WET_BULB_DIFF'] = result['T2M'] - result['T2MWET']
    result['PRESSURE_TENDENCY'] = result['PS'].diff()

    # Variables temporelles cycliques
    result['MONTH_SIN'] = np.sin(2 * np.pi * result['MO'] / 12)
    result['MONTH_COS'] = np.cos(2 * np.pi * result['MO'] / 12)
    result['DAY_SIN'] = np.sin(2 * np.pi * result['DY'] / 31)
    result['DAY_COS'] = np.cos(2 * np.pi * result['DY'] / 31)

    # Saisons
    result['SEASON'] = result['MO']
    
    return result

def explore_quantitative_variables(df):
    """Crée l'exploration des variables quantitatives"""
    quantitative_vars = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            quantitative_vars.append(col)

    if not quantitative_vars:
        return None

    n_vars = len(quantitative_vars)
    cols = min(3, n_vars)
    rows = (n_vars + cols - 1) // cols

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=quantitative_vars,
        specs=[[{"type": "histogram"}] * cols for _ in range(rows)]
    )

    for i, var in enumerate(quantitative_vars):
        row = i // cols + 1
        col = i % cols + 1

        var_data = df[var].dropna()
        min_val = var_data.min()
        max_val = var_data.max()

        fig.add_trace(
            go.Histogram(
                x=var_data,
                name=var,
                showlegend=False,
                nbinsx=30,
                marker_color='steelblue'
            ),
            row=row, col=col
        )

        mean_val = var_data.mean()
        std_val = var_data.std()
        median_val = var_data.median()

        fig.add_annotation(
            text=f"μ={mean_val:.2f}<br>σ={std_val:.2f}<br>Med={median_val:.2f}",
            xref=f"x{i+1}", yref=f"y{i+1}",
            x=0.7, y=0.9, xanchor="left", yanchor="top",
            showarrow=False, font=dict(size=9),
            row=row, col=col
        )

    fig.update_layout(
        height=300 * rows,
        title_text="Distribution des Variables Quantitatives",
        title_x=0.5,
        template="plotly_white"
    )

    return fig

def analyze_target_distribution(df, target_col='Target', date_col='DATE', ref=40):
    """Analyse la distribution de la variable cible"""
    target_data = df[target_col].dropna()

    Q1 = target_data.quantile(0.25)
    Q3 = target_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = target_data[(target_data < lower_bound) | (target_data > upper_bound)]
    n_outliers = len(outliers)
    pct_outliers = (n_outliers / len(target_data)) * 100

    above_ref = target_data[target_data > ref]
    n_above_ref = len(above_ref)
    pct_above_ref = (n_above_ref / len(target_data)) * 100

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[f'Distribution de {target_col}', 'Boxplot', 'Évolution temporelle'],
        row_heights=[0.4, 0.2, 0.4],
        vertical_spacing=0.08
    )

    fig.add_trace(
        go.Histogram(x=target_data, nbinsx=30, marker_color='steelblue', name='Distribution'),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(x=target_data, marker_color='lightblue', name='Boxplot'),
        row=2, col=1
    )

    df_clean = df[[date_col, target_col]].dropna()
    df_clean[date_col] = pd.to_datetime(df_clean[date_col])
    df_sorted = df_clean.sort_values(date_col)

    fig.add_trace(
        go.Scatter(x=df_sorted[date_col], y=df_sorted[target_col],
                   mode='lines+markers', marker_color='green', name='Évolution'),
        row=3, col=1
    )

    fig.add_hline(y=ref, line_dash="dash", line_color="red", row=3, col=1)

    fig.update_layout(
        height=800,
        template="plotly_white",
        showlegend=False,
        title_text=(
            f"Analyse de {target_col} - Outliers: {n_outliers} ({pct_outliers:.1f}%)"
            f" | Seuil >{ref}: {n_above_ref} ({pct_above_ref:.1f}%)"
        )
    )

    return fig, n_outliers, pct_outliers, n_above_ref, pct_above_ref

# Interface utilisateur
st.title("🌧️ Dashboard - Prédiction des Précipitations Corrigées")
# Navigation
st.sidebar.markdown("---")
st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Choisir une section:",
    ["📊 Vue d'ensemble", "🔍 Analyse Exploratoire", "🔗 Corrélations", 
        "🤖 Modélisation", "🎯 Prédictions"]
)

# Upload des fichiers
st.sidebar.title("📁 Upload des Données")
st.sidebar.markdown("---")

# Upload des fichiers CSV
train_file = st.sidebar.file_uploader("📤 Upload Train_data.csv", type=['csv'])
test_file = st.sidebar.file_uploader("📤 Upload Test_data.csv", type=['csv'])

# Chargement des données
if train_file is not None:
    st.session_state.data_loaded = True
    st.session_state.train_file = train_file
    if test_file is not None:
        st.session_state.test_file = test_file
    
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.session_state.data_loaded:
    # Chargement et traitement des données
    with st.spinner("🔄 Chargement et traitement des données..."):
        try:
            # Lire les fichiers CSV
            train_data = pd.read_csv(st.session_state.train_file)
            st.session_state.train_data = train_data  # Sauvegarder dans session_state
            
            if 'test_file' in st.session_state:
                test_data = pd.read_csv(st.session_state.test_file)
                st.session_state.test_data = test_data  # Sauvegarder dans session_state
            
            # Appliquer le feature engineering
            processed_data = generate_weather_features(train_data)
            
            # Filtrage des outliers comme dans le notebook (Target <= 50)
            processed_data = processed_data.query('Target<=50')
            st.session_state.processed_data = processed_data  # Sauvegarder dans session_state
            
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement des données: {str(e)}")
            st.stop()
    
    # Récupérer les données depuis session_state
    train_data = st.session_state.train_data
    processed_data = st.session_state.processed_data
    if 'test_data' in st.session_state:
        test_data = st.session_state.test_data
    
    st.success("✅ Données chargées avec succès!")
    
    
    # PAGE 1: Vue d'ensemble
    if page == "📊 Vue d'ensemble":
        st.markdown('<div class="highlight-box"><h2>📊 Vue d\'ensemble des Données</h2></div>', unsafe_allow_html=True)
        
        col2, col3, col4 = st.columns(3)
        
        
        
        with col2:
            st.metric(
                label="📅 Période couverte",
                value=f"{(processed_data['DATE'].max() - processed_data['DATE'].min()).days} jours"
            )
        
        with col3:
            st.metric(
                label="🌧️ Précipitation moyenne",
                value=f"{processed_data['Target'].mean():.2f} mm"
            )
        
        with col4:
            st.metric(
                label="📊 Variables créées",
                value="15+",
                delta="Features engineerées"
            )
        
        st.markdown("---")
        
        # Tableau de données
        st.subheader("🗂️ Aperçu des Données")
        st.dataframe(st.session_state.processed_data.head(100), use_container_width=True)
        
        
        # Informations sur les variables
        st.subheader("📋 Dictionnaire des Variables")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Variables Météorologiques de Base:**
            - **WS2M**: Vitesse du vent à 2m (m/s)
            - **T2M**: Température à 2m (°C)
            - **T2MDEW**: Température du point de rosée (°C)
            - **T2MWET**: Température du bulbe humide (°C)
            - **RH2M**: Humidité relative à 2m (%)
            - **PS**: Pression de surface (kPa)
            - **QV2M**: Humidité spécifique (g/kg)
            """)
        
        with col2:
            st.markdown("""
            **Variables Dérivées Créées:**
            - **HUMIDITY_SATURATION**: Ratio de saturation
            - **DEW_POINT_SPREAD**: Écart au point de rosée
            - **WET_BULB_DIFF**: Différence bulbe humide
            - **MONTH_SIN/COS**: Composantes cycliques mensuelles
            - **DAY_SIN/COS**: Composantes cycliques journalières
            - **SEASON**: Classification saisonnière
            - **PRESSURE_TENDENCY**: Tendance de pression
            """)
    
    # PAGE 2: Analyse Exploratoire
    elif page == "🔍 Analyse Exploratoire":
        st.markdown('<div class="highlight-box"><h2>🔍 Analyse Exploratoire des Données</h2></div>', unsafe_allow_html=True)
        
        # Distribution des variables quantitatives
        st.subheader("📊 Distribution des Variables Quantitatives")
        
        with st.spinner("Génération des graphiques..."):
            fig_quant = explore_quantitative_variables(st.session_state.processed_data)
            if fig_quant:
                st.plotly_chart(fig_quant, use_container_width=True)
        
        st.markdown("---")
        
        # Analyse de la variable cible
        st.subheader("🎯 Analyse de la Variable Cible (Précipitations)")
        
        ref_threshold = st.slider("Seuil de référence (mm)", 10, 50, 30)
        
        with st.spinner("Analyse de la variable cible..."):
            fig_target, n_outliers, pct_outliers, n_above_ref, pct_above_ref = analyze_target_distribution(
                st.session_state.processed_data, 'Target', 'DATE', ref_threshold)
            st.plotly_chart(fig_target, use_container_width=True)
        
        # Métriques de la variable cible
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🚨 Outliers détectés", f"{n_outliers}", f"{pct_outliers:.1f}%")
        
        with col2:
            st.metric(f"☔ Au-dessus de {ref_threshold}mm", f"{n_above_ref}", f"{pct_above_ref:.1f}%")
        
        with col3:
            st.metric("📏 Écart-type", f"{st.session_state.processed_data['Target'].std():.2f} mm")
    
    # PAGE 3: Corrélations
    elif page == "🔗 Corrélations":
        st.markdown('<div class="highlight-box"><h2>🔗 Analyse des Corrélations</h2></div>', unsafe_allow_html=True)
        
        # Matrice de corrélation
        st.subheader("🌡️ Matrice de Corrélation")
        
        cols_corr = [col for col in st.session_state.processed_data.select_dtypes(include=['float64', 'int64']).columns 
                     if col not in ['DATE', 'day', 'month', 'weekday']]
        
        corr_matrix = st.session_state.processed_data[cols_corr].corr().round(2)
        
        fig_corr = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu_r',
            showscale=True,
            annotation_text=corr_matrix.values.astype(str)
        )
        
        fig_corr.update_layout(
            title="Matrice de corrélation entre les variables",
            font=dict(size=12),
            height=700,
            template="plotly_white"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Top corrélations avec la cible
        st.subheader("🎯 Variables les Plus Corrélées avec les Précipitations")
        
        target_corr = corr_matrix['Target'].abs().sort_values(ascending=False)[1:11]  # Top 10 sans Target
        
        fig_top_corr = go.Figure(go.Bar(
            x=target_corr.values,
            y=target_corr.index,
            orientation='h',
            marker_color='steelblue'
        ))
        
        fig_top_corr.update_layout(
            title="Top 10 des Variables Corrélées avec les Précipitations",
            xaxis_title="Corrélation Absolue",
            yaxis_title="Variables",
            template="plotly_white",
            height=500
        )
        
        st.plotly_chart(fig_top_corr, use_container_width=True)
    
    # PAGE 4: Modélisation
    elif page == "🤖 Modélisation":
        st.markdown('<div class="highlight-box"><h2>🤖 Modélisation et Performance</h2></div>', unsafe_allow_html=True)
        
        # Sélection des variables pour la modélisation
        categorical_vars = ['SEASON']
        numerical_vars = ['WS2M', 'T2M', 'RH2M', 'PS', 'QV2M', 'MONTH_SIN', 'MONTH_COS']
        
        # Préparation des données
        X = st.session_state.processed_data[categorical_vars + numerical_vars]
        y = st.session_state.processed_data['Target']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Préprocesseur
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numerical_vars),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars)
            ]
        )
        
        # Modèles
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Entraînement et évaluation
        results = []
        trained_pipelines = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(models.items()):
            status_text.text(f'Entraînement du modèle: {name}...')
            
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', model)
            ])
            
            pipeline.fit(X_train, y_train)
            trained_pipelines[name] = pipeline  # Stocker le pipeline entraîné
            
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            results.append({
                'Modèle': name,
                'R² Train': r2_score(y_train, y_pred_train),
                'R² Test': r2_score(y_test, y_pred_test),
                'RMSE Train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'RMSE Test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAE Train': mean_absolute_error(y_train, y_pred_train),
                'MAE Test': mean_absolute_error(y_test, y_pred_test)
            })
            
            progress_bar.progress((i + 1) / len(models))
        
        status_text.text('✅ Entraînement terminé!')
        
        # Résultats
        results_df = pd.DataFrame(results)
        st.subheader("📊 Performance des Modèles")
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Graphique de comparaison
        fig_models = go.Figure()
        
        fig_models.add_trace(go.Bar(
            name='R² Test',
            x=results_df['Modèle'],
            y=results_df['R² Test'],
            marker_color='steelblue'
        ))
        
        fig_models.update_layout(
            title='Comparaison des Performances (R² Test)',
            xaxis_title='Modèles',
            yaxis_title='R² Score',
            template='plotly_white',
            height=500
        )
        
        st.plotly_chart(fig_models, use_container_width=True)
        
        # Identifier et sauvegarder le meilleur modèle
        best_model_idx = results_df['R² Test'].idxmax()
        best_model_name = results_df.loc[best_model_idx, 'Modèle']
        best_r2 = results_df.loc[best_model_idx, 'R² Test']
        best_pipeline = trained_pipelines[best_model_name]
        
        # Sauvegarder le meilleur modèle dans la session
        st.session_state.best_model = best_pipeline
        st.session_state.best_model_name = best_model_name
        st.session_state.best_model_r2 = best_r2
        
        # NOUVELLE SECTION: Importance des Variables
        st.markdown("---")
        st.subheader("📊 Importance des Variables par Modèle")
        st.write("Cette section montre quelles variables sont les plus importantes pour chaque modèle dans leurs prédictions.")
        
        # Récupérer les noms des features après preprocessing
        # Pour cela, on va utiliser un des pipelines entraînés
        sample_pipeline = trained_pipelines[list(trained_pipelines.keys())[0]]
        
        # Obtenir les noms des features après preprocessing
        feature_names_num = numerical_vars
        try:
            feature_names_cat = list(sample_pipeline.named_steps['preprocessor']
                                   .named_transformers_['cat']
                                   .get_feature_names_out(categorical_vars))
        except:
            feature_names_cat = [f"{var}_{i}" for var in categorical_vars for i in range(2)]  # Fallback
        
        all_feature_names = feature_names_num + feature_names_cat
        
        # Créer les graphiques d'importance pour chaque modèle
        n_models = len(trained_pipelines)
        cols_per_row = 2
        rows = (n_models + cols_per_row - 1) // cols_per_row
        
        fig_importance = make_subplots(
            rows=rows, cols=cols_per_row,
            subplot_titles=list(trained_pipelines.keys()),
            specs=[[{"type": "bar"}] * cols_per_row for _ in range(rows)]
        )
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (model_name, pipeline) in enumerate(trained_pipelines.items()):
            row = i // cols_per_row + 1
            col = i % cols_per_row + 1
            
            # Extraire l'importance selon le type de modèle
            regressor = pipeline.named_steps['regressor']
            
            if hasattr(regressor, 'feature_importances_'):
                # Pour Random Forest, Gradient Boosting
                importance = regressor.feature_importances_
            elif hasattr(regressor, 'coef_'):
                # Pour Linear Regression, Ridge
                importance = np.abs(regressor.coef_)
            else:
                # Fallback
                importance = np.ones(len(all_feature_names))
            
            # Assurer que nous avons le bon nombre de features
            if len(importance) != len(all_feature_names):
                # Ajuster si nécessaire
                min_len = min(len(importance), len(all_feature_names))
                importance = importance[:min_len]
                current_feature_names = all_feature_names[:min_len]
            else:
                current_feature_names = all_feature_names
            
            # Trier par importance décroissante et prendre le top 10
            indices = np.argsort(importance)[::-1][:10]
            top_importance = importance[indices]
            top_features = [current_feature_names[j] for j in indices]
            
            # Ajouter le graphique
            fig_importance.add_trace(
                go.Bar(
                    x=top_importance,
                    y=top_features,
                    orientation='h',
                    marker_color=colors[i % len(colors)],
                    name=model_name,
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Mettre à jour les axes
            fig_importance.update_xaxes(title_text="Importance", row=row, col=col)
            fig_importance.update_yaxes(title_text="Variables", row=row, col=col)
        
        # Mise en forme générale
        fig_importance.update_layout(
            height=400 * rows,
            title_text="Top 10 des Variables les Plus Importantes par Modèle",
            title_x=0.5,
            template="plotly_white",
            showlegend=False
        )
        
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Table détaillée de l'importance pour le meilleur modèle
        st.subheader(f"🏆 Détail d'Importance - {best_model_name}")
        
        best_pipeline = trained_pipelines[best_model_name]
        best_regressor = best_pipeline.named_steps['regressor']
        
        if hasattr(best_regressor, 'feature_importances_'):
            best_importance = best_regressor.feature_importances_
        elif hasattr(best_regressor, 'coef_'):
            best_importance = np.abs(best_regressor.coef_)
        else:
            best_importance = np.ones(len(all_feature_names))
        
        # Créer DataFrame avec importance
        if len(best_importance) == len(all_feature_names):
            importance_df = pd.DataFrame({
                'Variable': all_feature_names,
                'Importance': best_importance,
                'Importance_Pct': (best_importance / best_importance.sum()) * 100
            }).sort_values('Importance', ascending=False)
            
            # Ajouter interprétation des variables
            importance_df['Type'] = importance_df['Variable'].apply(
                lambda x: 'Météorologique' if x in numerical_vars 
                else 'Temporelle' if 'MONTH' in x or 'DAY' in x
                else 'Contextuelle'
            )
            
            st.dataframe(
                importance_df.round(4).head(15), 
                use_container_width=True,
                column_config={
                    "Variable": "🔧 Variable",
                    "Importance": st.column_config.ProgressColumn(
                        "📊 Importance",
                        help="Importance relative de la variable",
                        min_value=0,
                        max_value=importance_df['Importance'].max(),
                    ),
                    "Importance_Pct": "📈 Pourcentage (%)",
                    "Type": "📂 Catégorie"
                }
            )
            
            # Synthèse par type de variable
            type_summary = importance_df.groupby('Type')['Importance_Pct'].sum().sort_values(ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_type = go.Figure(data=[
                    go.Pie(
                        labels=type_summary.index,
                        values=type_summary.values,
                        hole=.3,
                        marker_colors=['#ff9999', '#66b3ff', '#99ff99']
                    )
                ])
                fig_type.update_layout(
                    title="Contribution par Type de Variable",
                    height=400,
                    template="plotly_white"
                )
                st.plotly_chart(fig_type, use_container_width=True)
            
            with col2:
                st.markdown("### 💡 Interprétation")
                
                top_var = importance_df.iloc[0]['Variable']
                top_type = importance_df.iloc[0]['Type']
                top_pct = importance_df.iloc[0]['Importance_Pct']
                
                st.markdown(f"""
                **Variable la plus importante:** `{top_var}`
                
                - **Type:** {top_type}
                - **Contribution:** {top_pct:.1f}%
                
                **Répartition par catégorie:**
                """)
                
                for var_type, pct in type_summary.items():
                    emoji = {"Météorologique": "🌡️", "Temporelle": "📅", "Contextuelle": "🗂️"}
                    st.markdown(f"- {emoji.get(var_type, '📊')} **{var_type}:** {pct:.1f}%")
        
        st.markdown("---")
        
        # Entraîner le meilleur modèle sur toutes les données d'entraînement
        X_full = st.session_state.processed_data[categorical_vars + numerical_vars]
        y_full = st.session_state.processed_data['Target']
        
        final_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', models[best_model_name])
        ])
        
        with st.spinner(f"🔄 Entraînement final du meilleur modèle {best_model_name}..."):
            final_model.fit(X_full, y_full)
        
        # Sauvegarder le modèle final par défaut (meilleur modèle)
        if 'final_model' not in st.session_state:
            st.session_state.final_model = final_model
        
        # Exporter le modèle
        model_filename = f"best_model_{best_model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(final_model, model_filename)
        
        st.success(f"🏆 Meilleur modèle: **{best_model_name}** (R² = {best_r2:.4f})")
        st.success(f"💾 Modèles entraînés et prêts à être utilisés")
        
        # Section de sélection et téléchargement des modèles
        st.markdown("---")
        st.subheader("📥 Sélection et Téléchargement de Modèle")
        
        # Message d'information
        st.info("""
        💡 **Nouvelle fonctionnalité :** Vous pouvez maintenant choisir :
        - **Quel modèle télécharger** (pour usage externe)  
        - **Quel modèle utiliser** pour les prédictions dans ce dashboard
        
        Cela vous permet d'expérimenter avec différents modèles selon vos besoins !
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sélection du modèle à télécharger
            selected_model_download = st.selectbox(
                "🔽 Choisir le modèle à télécharger:",
                options=list(trained_pipelines.keys()),
                index=list(trained_pipelines.keys()).index(best_model_name),
                help="Sélectionnez le modèle que vous voulez télécharger"
            )
            
            # Afficher les performances du modèle sélectionné
            selected_perf = results_df[results_df['Modèle'] == selected_model_download].iloc[0]
            st.info(f"**Performance:** R² = {selected_perf['R² Test']:.4f} | RMSE = {selected_perf['RMSE Test']:.3f}")
        
        with col2:
            # Sélection du modèle pour les prédictions
            selected_model_prediction = st.selectbox(
                "🎯 Choisir le modèle pour les prédictions:",
                options=list(trained_pipelines.keys()),
                index=list(trained_pipelines.keys()).index(best_model_name),
                help="Ce modèle sera utilisé dans la section Prédictions"
            )
            
            # Sauvegarder le modèle sélectionné pour les prédictions
            st.session_state.selected_prediction_model = selected_model_prediction
            st.session_state.selected_prediction_pipeline = trained_pipelines[selected_model_prediction]
            
            # Afficher les performances du modèle sélectionné pour prédiction
            pred_perf = results_df[results_df['Modèle'] == selected_model_prediction].iloc[0]
            st.success(f"**Performance:** R² = {pred_perf['R² Test']:.4f} | RMSE = {pred_perf['RMSE Test']:.3f}")
        
        # Entraîner le modèle sélectionné pour téléchargement sur toutes les données
        if st.button("🔄 Préparer le modèle sélectionné pour téléchargement"):
            with st.spinner(f"🔄 Entraînement final du modèle {selected_model_download} sur toutes les données..."):
                # Entraîner sur toutes les données d'entraînement
                X_full = st.session_state.processed_data[categorical_vars + numerical_vars]
                y_full = st.session_state.processed_data['Target']
                
                final_model_download = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', models[selected_model_download])
                ])
                
                final_model_download.fit(X_full, y_full)
                
                # Sauvegarder le modèle final pour téléchargement
                st.session_state.final_model_download = final_model_download
                st.session_state.download_model_name = selected_model_download
                
                st.success(f"✅ Modèle {selected_model_download} prêt pour téléchargement!")
        
                    # Téléchargement du modèle préparé
        if 'final_model_download' in st.session_state:
            download_model_name = st.session_state.download_model_name
            model_filename = f"model_{download_model_name.lower().replace(' ', '_')}.pkl"
            
            # Exporter le modèle
            joblib.dump(st.session_state.final_model_download, model_filename)
            
            # Bouton de téléchargement
            with open(model_filename, 'rb') as f:
                st.download_button(
                    label=f"📥 Télécharger {download_model_name}",
                    data=f.read(),
                    file_name=model_filename,
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        # Résumé des sélections
        st.markdown("---")
        st.subheader("📋 Résumé des Sélections")
        
        col1, col2 = st.columns(2)
        with col1:
            download_model = st.session_state.get('download_model_name', selected_model_download)
            download_perf = results_df[results_df['Modèle'] == download_model].iloc[0] if download_model in results_df['Modèle'].values else None
            
            if download_perf is not None:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                            padding: 1rem; border-radius: 0.5rem; color: white;">
                    <h4 style="margin: 0; color: white;">📥 Modèle pour Téléchargement</h4>
                    <p style="margin: 0.5rem 0;"><strong>{download_model}</strong></p>
                    <small>R² = {download_perf['R² Test']:.4f} | RMSE = {download_perf['RMSE Test']:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            pred_model = st.session_state.get('selected_prediction_model', selected_model_prediction)
            pred_perf = results_df[results_df['Modèle'] == pred_model].iloc[0] if pred_model in results_df['Modèle'].values else None
            
            if pred_perf is not None:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%); 
                            padding: 1rem; border-radius: 0.5rem; color: white;">
                    <h4 style="margin: 0; color: white;">🎯 Modèle pour Prédictions</h4>
                    <p style="margin: 0.5rem 0;"><strong>{pred_model}</strong></p>
                    <small>R² = {pred_perf['R² Test']:.4f} | RMSE = {pred_perf['RMSE Test']:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Entraîner aussi le modèle sélectionné pour prédictions (pour usage immédiat)
        with st.spinner(f"🔄 Préparation du modèle {selected_model_prediction} pour les prédictions..."):
            X_full = st.session_state.processed_data[categorical_vars + numerical_vars]
            y_full = st.session_state.processed_data['Target']
            
            final_model_prediction = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', models[selected_model_prediction])
            ])
            
            final_model_prediction.fit(X_full, y_full)
            
            # Sauvegarder le modèle final pour prédictions
            st.session_state.final_model = final_model_prediction
    
    # PAGE 5: Prédictions
    elif page == "🎯 Prédictions":
        st.markdown('<div class="highlight-box"><h2>🎯 Prédictions et Soumission</h2></div>', unsafe_allow_html=True)
        
        # Vérifier si les données sont chargées
        if 'processed_data' not in st.session_state:
            st.warning("⚠️ Aucune donnée trouvée. Veuillez d'abord uploader vos données dans la section **📊 Vue d'ensemble**.")
            st.info("👈 Utilisez la navigation dans la barre latérale pour uploader vos fichiers CSV.")
            
        
        # Vérifier si un modèle a été entraîné
        if 'final_model' not in st.session_state:
            st.warning("⚠️ Aucun modèle entraîné trouvé. Veuillez d'abord aller dans la section **🤖 Modélisation** pour entraîner les modèles.")
            st.info("👈 Utilisez la navigation dans la barre latérale pour accéder à la section Modélisation.")
            
        
        # Afficher les informations du modèle utilisé
        if 'selected_prediction_model' in st.session_state:
            model_name = st.session_state.selected_prediction_model
            # Récupérer les performances du modèle depuis les résultats
            if 'processed_data' in st.session_state:
                # Afficher le modèle sélectionné
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                            padding: 1rem; border-radius: 0.5rem; color: white; margin: 1rem 0;">
                    <strong>🤖 Modèle utilisé pour les prédictions:</strong> {model_name}
                    <br><small>💡 Modèle sélectionné dans la section Modélisation</small>
                </div>
                """, unsafe_allow_html=True)
        elif 'best_model_name' in st.session_state:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); 
                        padding: 1rem; border-radius: 0.5rem; color: white; margin: 1rem 0;">
                <strong>🤖 Modèle utilisé:</strong> {st.session_state.best_model_name} 
                (R² = {st.session_state.best_model_r2:.4f})
                <br><small>💡 Meilleur modèle automatiquement sélectionné</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Option pour changer de modèle directement depuis cette page
        if 'processed_data' in st.session_state:
            st.markdown("### 🔄 Changer de Modèle (Optionnel)")
            
            # Récupérer les modèles disponibles (s'ils existent)
            available_models = []
            if 'final_model' in st.session_state:
                available_models = ['Linear Regression', 'Ridge', 'Random Forest', 'Gradient Boosting']
            
            if available_models:
                current_model = st.session_state.get('selected_prediction_model', st.session_state.get('best_model_name', 'Gradient Boosting'))
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    new_model = st.selectbox(
                        "Sélectionner un autre modèle:",
                        options=available_models,
                        index=available_models.index(current_model) if current_model in available_models else 0,
                        help="Changez de modèle pour voir différentes prédictions"
                    )
                
                with col2:
                    if st.button("🔄 Appliquer le changement"):
                        # Réentraîner le nouveau modèle sélectionné
                        categorical_vars = ['SEASON']
                        numerical_vars = ['WS2M', 'T2M', 'RH2M', 'PS', 'QV2M', 'MONTH_SIN', 'MONTH_COS']
                        
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', RobustScaler(), numerical_vars),
                                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_vars)
                            ]
                        )
                        
                        models_dict = {
                            'Linear Regression': LinearRegression(),
                            'Ridge': Ridge(alpha=1.0),
                            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
                        }
                        
                        with st.spinner(f"🔄 Changement vers {new_model}..."):
                            X_full = st.session_state.processed_data[categorical_vars + numerical_vars]
                            y_full = st.session_state.processed_data['Target']
                            
                            new_final_model = Pipeline([
                                ('preprocessor', preprocessor),
                                ('regressor', models_dict[new_model])
                            ])
                            
                            new_final_model.fit(X_full, y_full)
                            
                            # Mettre à jour le modèle dans la session
                            st.session_state.final_model = new_final_model
                            st.session_state.selected_prediction_model = new_model
                            
                            st.success(f"✅ Modèle changé vers {new_model}!")
                            st.rerun()
            
            st.markdown("---")
        
        # Section 1: Prédictions sur le fichier de test (si disponible)
        if 'test_data' in st.session_state:
            st.subheader("🧪 Prédictions sur les Données de Test")
            
            # Utiliser le modèle final sauvegardé
            final_model = st.session_state.final_model
            
            # Prédictions sur test
            test_data = st.session_state.test_data
            test_data_processed = generate_weather_features(test_data)
            
            # Variables utilisées pour la prédiction
            categorical_vars = ['SEASON']
            numerical_vars = ['WS2M', 'T2M', 'RH2M', 'PS', 'QV2M', 'MONTH_SIN', 'MONTH_COS']
            X_test_final = test_data_processed[categorical_vars + numerical_vars]
            
            with st.spinner("🔮 Génération des prédictions avec le meilleur modèle..."):
                predictions = final_model.predict(X_test_final)
            
            # Créer le fichier de soumission
            test_data = st.session_state.test_data
            if 'ID' in test_data.columns:
                submission = pd.DataFrame({
                    'ID': test_data['ID'],
                    'Target': predictions
                })
            else:
                submission = pd.DataFrame({
                    'Target': predictions
                })
            
            st.success(f"✅ Prédictions générées pour {len(predictions):,} observations")
            
            # Afficher les statistiques des prédictions
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Moyenne", f"{predictions.mean():.2f} mm")
            with col2:
                st.metric("📈 Maximum", f"{predictions.max():.2f} mm")
            with col3:
                st.metric("📉 Minimum", f"{predictions.min():.2f} mm")
            with col4:
                st.metric("📏 Écart-type", f"{predictions.std():.2f} mm")
            
            # Graphique des prédictions
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Histogram(
                x=predictions,
                nbinsx=50,
                marker_color='steelblue',
                name='Prédictions'
            ))
            fig_pred.update_layout(
                title='Distribution des Prédictions sur les Données de Test',
                xaxis_title='Précipitations prédites (mm)',
                yaxis_title='Fréquence',
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Téléchargement du fichier de soumission
            model_name_for_file = st.session_state.get('selected_prediction_model', 
                                                       st.session_state.get('best_model_name', 'model'))
            csv = submission.to_csv(index=False)
            st.download_button(
                label="📥 Télécharger le fichier de soumission",
                data=csv,
                file_name=f"predictions_{model_name_for_file.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
            # Aperçu des prédictions
            st.subheader("👀 Aperçu des Prédictions")
            st.dataframe(submission.head(100), use_container_width=True)
            
            st.markdown("---")
        
        # Section 2: Simulateur interactif
        st.subheader("🌦️ Simulateur de Prédiction Interactif")
        st.write("Ajustez les paramètres météorologiques pour voir une prédiction:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            temp = st.slider("🌡️ Température (°C)", 15.0, 35.0, 25.0, 0.5)
            humidity = st.slider("💧 Humidité relative (%)", 30.0, 100.0, 70.0, 1.0)
            wind = st.slider("💨 Vitesse du vent (m/s)", 0.0, 10.0, 3.0, 0.1)
        
        with col2:
            pressure = st.slider("📊 Pression (kPa)", 95.0, 108.0, 101.3, 0.1)
            specific_humidity = st.slider("💨 Humidité spécifique (g/kg)", 5.0, 20.0, 12.0, 0.5)
            month = st.slider("📅 Mois", 1, 12, 6)
        
        with col3:
            season = st.selectbox("🌍 Saison Congo-Brazzaville", [1, 2, 3, 4], 1,
                                 format_func=lambda x: {1: "Petite saison sèche (jan-fév)", 
                                                       2: "Grande saison sèche (juin-sept)", 
                                                       3: "Grande saison pluies (oct-déc)", 
                                                       4: "Petite saison pluies (mars-mai)"}[x])
        
        # Calcul des variables dérivées
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        
        # Création du vecteur de prédiction
        prediction_data = pd.DataFrame({
            'WS2M': [wind],
            'T2M': [temp],
            'RH2M': [humidity],
            'PS': [pressure],
            'QV2M': [specific_humidity],
            'MONTH_SIN': [month_sin],
            'MONTH_COS': [month_cos],
            'SEASON': [season]
        })
        
        # Utiliser le modèle final sauvegardé
        final_model = st.session_state.final_model
        prediction = final_model.predict(prediction_data)[0]
        
        # Affichage de la prédiction
        col1, col2, col3 = st.columns([1, 2, 1])
        
        # Récupérer le nom du modèle utilisé
        model_display_name = st.session_state.get('selected_prediction_model', 
                                                  st.session_state.get('best_model_name', 'Modèle'))
        
        with col2:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 2rem;
                border-radius: 1rem;
                text-align: center;
                color: white;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                margin: 2rem 0;
            ">
                <h2 style="margin: 0; color: white;">🌧️ Prédiction</h2>
                <h1 style="margin: 0.5rem 0; font-size: 3rem; color: white;">{prediction:.2f} mm</h1>
                <p style="margin: 0; opacity: 0.9;">Précipitations attendues</p>
                <small style="opacity: 0.8;">Modèle: {model_display_name}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Interprétation
        if prediction < 1:
            interpretation = "☀️ Temps sec attendu"
            color = "#f39c12"
        elif prediction < 10:
            interpretation = "🌤️ Précipitations légères"
            color = "#3498db"
        elif prediction < 25:
            interpretation = "🌧️ Précipitations modérées"
            color = "#2980b9"
        else:
            interpretation = "⛈️ Précipitations importantes"
            color = "#8e44ad"
        
        st.markdown(f"""
        <div style="
            background-color: {color};
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            color: white;
            margin: 1rem 0;
        ">
            <h3 style="margin: 0; color: white;">{interpretation}</h3>
        </div>
        """, unsafe_allow_html=True)

else:
    # Page d'accueil si pas de données
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h1>🌧️ Bienvenue dans le Dashboard de Prédiction des Précipitations</h1>
        <p style="font-size: 1.2rem; color: #666;">
            Ce dashboard utilise des techniques avancées de machine learning pour prédire 
            les précipitations à partir de variables météorologiques.
        </p>
        <p style="font-size: 1.1rem;">
            👈 Uploadez vos fichiers <strong>Train_data.csv</strong> et <strong>Test_data.csv</strong> dans la barre latérale pour commencer l'exploration.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Fonctionnalités du dashboard
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Vue d'ensemble
        - Statistiques descriptives
        - Aperçu des données
        - Dictionnaire des variables
        """)
    
    with col2:
        st.markdown("""
        ### 🔍 Analyse Exploratoire
        - Distribution des variables
        - Analyse de la variable cible
        - Détection d'outliers
        """)
    
    with col3:
        st.markdown("""
        ### 🔗 Corrélations
        - Matrice de corrélation
        - Variables les plus corrélées
        - Analyse des relations
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🤖 Modélisation
        - Comparaison de 4 algorithmes
        - Métriques de performance
        - Sélection du meilleur modèle
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Prédictions
        - Simulateur interactif
        - Prédictions en temps réel
        - Interprétation des résultats
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p>🌧️ Dashboard créé avec Streamlit | Données météorologiques | Machine Learning</p>
</div>
""", unsafe_allow_html=True)