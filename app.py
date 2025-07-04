import pandas as pd
import pickle
import plotly.express as px
import pydeck as pdk
import streamlit as st


@st.cache_data
def load_data(dists_filepath):
    orgs_df = pd.read_csv('orgs.csv')
    leaflets_df = pd.read_csv('leaflets.csv', sep='¦', encoding='latin-1', engine='python')
    dists_df = pd.read_csv(dists_filepath)

    orgs_df.rename(columns={'ID': 'Org_ID'}, inplace=True)

    dists_orgs_merged = orgs_df.merge(dists_df, on='Org_ID')

    earliest_delivery_date = pd.to_datetime(dists_df['Delivered_Date'].min())
    latest_delivery_date = pd.to_datetime(dists_df['Delivered_Date'].max())

    unique_org_types = orgs_df['Type'].unique()
    unique_cities = dists_df['Delivered_City'].unique()
    unique_neighbourhoods = dists_df['Delivered_Neighbourhood'].unique()

    min_age = int(dists_df['Addressee_Age'].min())
    max_age = int(dists_df['Addressee_Age'].max())

    min_income = int(dists_df['Annual_Income_SEK'].min())
    max_income = int(dists_df['Annual_Income_SEK'].max())

    min_house_price = int(dists_df['Property_Value_SEK'].min())
    max_house_price = int(dists_df['Property_Value_SEK'].max())

    max_cars = int(dists_df['Household_Cars'].max())
    max_occupants = int(dists_df['Household_Size'].max())

    return earliest_delivery_date, latest_delivery_date, unique_org_types, unique_cities, unique_neighbourhoods,\
        min_age, max_age, min_income, max_income, min_house_price, max_house_price, max_cars, max_occupants,\
        dists_orgs_merged, leaflets_df

st.write('DISCLAIMER! These data are FAKE, generated for the BALADRIA Summer School in Digital '
         'Humanities 2025. They are released under a CC-BY license, but should not be used for any "real" '
         'analysis. Any data seemingly relating to a real person are coincidental.')
st.divider()
st.title('"Samhällsinformation": Exploring persuasive communication in the wake of wildfires in Sweden')

earliest_delivery_date, latest_delivery_date, unique_org_types, unique_cities, unique_neighbourhoods, \
    min_age, max_age, min_income, max_income, min_house_price, max_house_price, max_cars, max_occupants, \
    dists_orgs_df, leaflets_df = load_data('dists.csv')

with st.sidebar:
    with st.expander(label='Filter data'):
        filter_org = st.multiselect(label='Select organisation type(s)',
                                    options=unique_org_types)

        filter_text = st.text_input(label='Text to search leaflets for')

        filter_city = st.multiselect(label='Select city/cities',
                                     options=unique_cities)

        filter_neighbourhood = st.multiselect(label='Select neighbourhood',
                                              options=unique_neighbourhoods)

        filter_dates = st.date_input(label='Select delivery dates',
                                     value=(earliest_delivery_date,
                                            latest_delivery_date),
                                     min_value=earliest_delivery_date,
                                     max_value=latest_delivery_date)

        filter_gender = st.radio(label='Select gender of recipients',
                                 options=['All', 'Female', 'Male', 'Other']
                                 )

        filter_age = st.slider(label='Age range of recipients',
                               value=[min_age, max_age],
                               min_value=min_age,
                               max_value=max_age)

        filter_income = st.slider(label='Annual income range of recipients (SEK)',
                                  value=[min_income, max_income],
                                  min_value=min_income,
                                  max_value=max_income)

        filter_house_price = st.slider(
            label='Price of property distributed to (SEK)',
            value=[min_house_price, max_house_price],
            min_value=min_house_price,
            max_value=max_house_price
        )

        filter_occupants = st.slider(
            label='Number of occupants in the household',
            value=[1, max_occupants],
            min_value=1,
            max_value=max_occupants
        )

        filter_cars = st.slider(
            label='Number of cars in household',
            value=[0, max_cars],
            min_value=0,
            max_value=max_cars
        )

        st.divider()

        st.write('The fake data were generated in several iterations, each more advanced and realistic than the last. '
                 'The default data displayed here are the latest (fifth) iteration. To view other iterations, select '
                 'them here.')
        data_iteration = st.selectbox(label='Select data iteration',
                     options=['Iteration 5 (latest)',
                     'Iteration 4',
                     'Iteration 3',
                     'Iteration 2',
                     'Iteration 1']
                     )

    with st.expander(label='Filtering options'):
        st.write('It is possible to include the title and text of each leaflet in the results. However, please note '
                 'that since the dataframe displayed is based on each unique distribution, this will cause an extremely '
                 'large amount of text to be displayed and may crash the programme. The estimated maximum number of '
                 'rows is 200,000.')
        show_leaflet_text = st.checkbox(label='Include leaflet text')
    with st.expander(label='Analysis'):
        analysis_type = st.radio(label='Select analysis type',
                                 options=['None (raw data)', 'Demographics', 'Sentiment', 'Topics', 'Geographical'])

try:
    if data_iteration:
        if data_iteration != 'Iteration 5 (latest)':
            num = data_iteration.split(' ')[1]
            earliest_delivery_date, latest_delivery_date, unique_org_types, unique_cities, unique_neighbourhoods, \
                min_age, max_age, min_income, max_income, min_house_price, max_house_price, max_cars, max_occupants, \
                dists_orgs_df, leaflets_df = load_data(f'old_datasets/dists_{num}.csv')
        else:
            earliest_delivery_date, latest_delivery_date, unique_org_types, unique_cities, unique_neighbourhoods, \
                min_age, max_age, min_income, max_income, min_house_price, max_house_price, max_cars, max_occupants, \
                dists_orgs_df, leaflets_df = load_data(f'dists.csv')
    filtered_df = dists_orgs_df

    if filter_org:
        filtered_df = filtered_df[filtered_df['Type'].isin(filter_org)]

    if filter_text:
        filtered_df = filtered_df[filtered_df['Leaflet_Text'].str.contains(filter_text, case=False, na=False)]

    start_date, end_date = filter_dates
    filtered_df = filtered_df[(pd.to_datetime(filtered_df['Delivered_Date']) >= pd.to_datetime(start_date)) &
                              (pd.to_datetime(filtered_df['Delivered_Date']) <= pd.to_datetime(end_date))]

    if filter_city:
        filtered_df = filtered_df[filtered_df['Delivered_City'].isin(filter_city)]

    if filter_neighbourhood:
        filtered_df = filtered_df[filtered_df['Delivered_Neighbourhood'].isin(filter_neighbourhood)]

    if filter_gender != 'All':
        filtered_df = filtered_df[filtered_df['Addressee_Gender'] == filter_gender]

    filtered_df = filtered_df[(filtered_df['Addressee_Age'] >= filter_age[0]) &
                               (filtered_df['Addressee_Age'] <= filter_age[1])]

    filtered_df = filtered_df[(filtered_df['Annual_Income_SEK'] >= filter_income[0]) &
                               (filtered_df['Annual_Income_SEK'] <= filter_income[1])]

    filtered_df = filtered_df[(filtered_df['Property_Value_SEK'] >= filter_house_price[0]) &
                               (filtered_df['Property_Value_SEK'] <= filter_house_price[1])]

    filtered_df = filtered_df[(filtered_df['Household_Size'] >= filter_occupants[0]) &
                               (filtered_df['Household_Size'] <= filter_occupants[1])]

    filtered_df = filtered_df[(filtered_df['Household_Cars'] >= filter_cars[0]) &
                               (filtered_df['Household_Cars'] <= filter_cars[1])]

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Distributions", f"{len(filtered_df):,}")
    col2.metric("Unique Organizations in Filter", f"{filtered_df['Org_ID'].nunique():,}")
    col3.metric("Average Recipient Age", f"{filtered_df['Addressee_Age'].mean():.1f}")
    if show_leaflet_text:
        filtered_df = filtered_df.merge(leaflets_df, on='Leaflet_ID')
    if analysis_type == 'None (raw data)':
        try:
            filtered_df = filtered_df.drop(columns=['Org_ID', 'Distribution_ID', 'Leaflet_ID', 'Latitude', 'Longitude'])
        except KeyError:
            try:
                filtered_df = filtered_df.drop(columns=['Org_ID', 'Distribution_ID', 'Leaflet_ID'])
            except KeyError:
                filtered_df = filtered_df.drop(columns=['Distribution_ID', 'Leaflet_ID'])
        if not show_leaflet_text:
            filtered_df = filtered_df.loc[:, ['Delivered_Date', 'Delivered_City', 'Delivered_Neighbourhood', 'Addressee_Name', 'Addressee_Gender',
                    'Addressee_Age', 'Annual_Income_SEK', 'Property_Value_SEK', 'Household_Cars', 'Household_Size',
                    'Name', 'Type']]
        else:
            filtered_df = filtered_df.loc[:,
                          ['Delivered_Date', 'Delivered_City', 'Delivered_Neighbourhood', 'Addressee_Name',
                           'Addressee_Gender',
                           'Addressee_Age', 'Annual_Income_SEK', 'Property_Value_SEK', 'Household_Cars',
                           'Household_Size',
                           'Name', 'Type', 'Title', 'Text']]
        st.dataframe(filtered_df)
    elif analysis_type == 'Demographics':
        st.header("In-depth Demographic Analysis")
        st.markdown("Explore the demographic characteristics of the recipients based on your master filter settings.")

        # Create sub-tabs for better organization
        sub_tab_geo, sub_tab_household, sub_tab_economic = st.tabs([
            "🗺️ Geographic Breakdown", "🏠 Household Profile", "💰 Economic Profile"
        ])

        with sub_tab_geo:
            st.subheader("Distribution by Location")

            # City-level distribution
            city_counts = filtered_df['Delivered_City'].value_counts()
            fig_city = px.bar(
                city_counts, x=city_counts.index, y=city_counts.values,
                title="Leaflet Distributions by City",
                labels={'x': 'City', 'y': 'Number of Distributions'}
            )
            st.plotly_chart(fig_city, use_container_width=True)

            # Interactive neighbourhood-level distribution
            st.markdown("---")
            st.subheader("Neighbourhood Drill-Down")

            # Only show cities that are in the filtered data
            available_cities = filtered_df['Delivered_City'].unique().tolist()

            selected_city = st.selectbox(
                "Choose a city to see its neighbourhood breakdown:",
                options=available_cities
            )

            if selected_city:
                df_city_specific = filtered_df[filtered_df['Delivered_City'] == selected_city]
                neighbourhood_counts = df_city_specific['Delivered_Neighbourhood'].value_counts()
                fig_hood = px.bar(
                    neighbourhood_counts, x=neighbourhood_counts.index, y=neighbourhood_counts.values,
                    title=f"Distributions by Neighbourhood in {selected_city}",
                    labels={'x': 'Neighbourhood', 'y': 'Number of Distributions'}
                )
                st.plotly_chart(fig_hood, use_container_width=True)

        with sub_tab_household:
            st.subheader("Recipient Household Characteristics")
            col1, col2, col3 = st.columns(3)

            with col1:
                gender_counts = filtered_df['Addressee_Gender'].value_counts()
                fig_gender = px.pie(gender_counts, values=gender_counts.values, names=gender_counts.index,
                                    title="Gender of Recipients")
                st.plotly_chart(fig_gender, use_container_width=True)

            with col2:
                size_counts = filtered_df['Household_Size'].value_counts().sort_index()
                fig_size = px.pie(size_counts, values=size_counts.values, names=size_counts.index,
                                  title="Household Size")
                st.plotly_chart(fig_size, use_container_width=True)

            with col3:
                car_counts = filtered_df['Household_Cars'].value_counts().sort_index()
                fig_cars = px.pie(car_counts, values=car_counts.values, names=car_counts.index,
                                  title="Cars per Household")
                st.plotly_chart(fig_cars, use_container_width=True)

        with sub_tab_economic:
            st.subheader("Recipient Economic Profiles")
            st.markdown(
                "These heatmaps show the concentration of recipients. Darker areas indicate more distributions.")

            # Age vs. Income
            if len(filtered_df) > 10:
                fig_density_age = px.density_heatmap(
                    filtered_df, x="Addressee_Age", y="Annual_Income_SEK",
                    nbinsx=30, nbinsy=30,
                    title="Density of Recipients by Age and Annual Income",
                    labels={'Addressee_Age': 'Recipient Age', 'Annual_Income_SEK': 'Annual Income (SEK)'}
                )
                st.plotly_chart(fig_density_age, use_container_width=True, key='ec_profile')

                # Income vs. Property Value
                st.markdown("---")
                fig_density_prop = px.density_heatmap(
                    filtered_df, x="Annual_Income_SEK", y="Property_Value_SEK",
                    nbinsx=30, nbinsy=30,
                    title="Density of Recipients by Income and Property Value",
                    labels={'Annual_Income_SEK': 'Annual Income (SEK)', 'Property_Value_SEK': 'Property Value (SEK)'}
                )
                st.plotly_chart(fig_density_prop, use_container_width=True, key='income_prop_val')
            else:
                st.warning("Not enough data points in the current filter to draw density maps.")
    elif analysis_type == 'Sentiment':
        sents_df = pd.read_csv('sents.csv')
        filtered_df = filtered_df.merge(sents_df, on='Leaflet_ID')

        st.header("Sentiment Analysis")
        sentiment_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
        sentiment_color_map = {'Positive': '#2ca02c', 'Neutral': '#ff7f0e', 'Negative': '#d62728'}
        sentiment_order = ['Positive', 'Neutral', 'Negative']

        df_plot = filtered_df.copy()
        df_plot['sentiment_label'] = df_plot['sent_rounded'].map(sentiment_map)

        st.markdown("#### Overall Sentiment Breakdown")
        sentiment_counts = df_plot['sentiment_label'].value_counts()
        fig_pie = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                         title="Proportion of Leaflets by Sentiment (in filtered data)", color=sentiment_counts.index,
                         color_discrete_map=sentiment_color_map)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### Sentiment by Top Organizations")
        col1, col2 = st.columns([1, 3])
        with col1:
            top_n_sentiment = st.slider("Select Top N Orgs to Compare", 3, 15, 5, key="sentiment_slider")
            normalize_view = st.checkbox("Show as Percentage (Normalized)", key="normalize_check")

        top_org_names = df_plot['Name'].value_counts().nlargest(top_n_sentiment).index
        df_top_orgs = df_plot[df_plot['Name'].isin(top_org_names)]

        histnorm_type = 'percent' if normalize_view else None
        title_text = f"Sentiment Breakdown for Top {top_n_sentiment} Orgs ({'Percentage' if normalize_view else 'Count'})"

        fig_org_sentiment = px.histogram(df_top_orgs, x='Name', color='sentiment_label',
                                         barmode='stack' if normalize_view else 'group', histnorm=histnorm_type,
                                         title=title_text,
                                         labels={'Name': 'Organization Name', 'sentiment_label': 'Sentiment'},
                                         category_orders={"Name": top_org_names, "sentiment_label": sentiment_order},
                                         color_discrete_map=sentiment_color_map)
        fig_org_sentiment.update_xaxes(categoryorder='total descending')
        st.plotly_chart(fig_org_sentiment, use_container_width=True)

        st.markdown("#### Sentiment vs. Recipient Demographics")
        col1, col2 = st.columns(2)
        with col1:
            fig_box_income = px.box(df_plot, x='sentiment_label', y='Annual_Income_SEK', color='sentiment_label',
                                    title="Recipient Income by Leaflet Sentiment",
                                    labels={'sentiment_label': 'Sentiment', 'Annual_Income_SEK': 'Annual Income (SEK)'},
                                    category_orders={"sentiment_label": sentiment_order},
                                    color_discrete_map=sentiment_color_map)
            st.plotly_chart(fig_box_income, use_container_width=True)
        with col2:
            fig_box_age = px.box(df_plot, x='sentiment_label', y='Addressee_Age', color='sentiment_label',
                                 title="Recipient Age by Leaflet Sentiment",
                                 labels={'sentiment_label': 'Sentiment', 'Addressee_Age': 'Recipient Age'},
                                 category_orders={"sentiment_label": sentiment_order},
                                 color_discrete_map=sentiment_color_map)
            st.plotly_chart(fig_box_age, use_container_width=True)
    elif analysis_type == 'Topics':
        with open('lda_model.pkl', 'rb') as f:
            lda_model = pickle.load(f)
        topics_df = pd.read_csv('topics.csv')
        filtered_df = filtered_df.merge(topics_df, on='Leaflet_ID')

        topic_names = {}
        for topic_id, topic_words in lda_model.show_topics(formatted=False, num_words=10):
            topic_names[topic_id] = ", ".join([word for word, _ in topic_words])
        filtered_df['dominant_topic_name'] = filtered_df['dominant_topic'].map(topic_names)

        st.header("Analysis of Leaflet Topics")

        sub_tab_explore, sub_tab_keywords, sub_tab_orgs = st.tabs([
            "🌐 Interactive Topic Explorer", "🔑 Topic Keywords", "🏢 Topics by Organization"
        ])

        with sub_tab_explore:
            st.subheader("Interactive LDA Visualization")
            st.markdown(
                "Explore the relationship between topics and their most relevant keywords. From the `pyLDAvis` library.")

            # Load and display the pre-generated HTML file
            with open('pyldavis_visualization.html', 'r') as f:
                html_string = f.read()
            st.components.v1.html(html_string, width=1300, height=800, scrolling=True)

        with sub_tab_keywords:
            st.subheader("Top Keywords for Each Topic")
            st.markdown("These are the most representative words for each of the topics identified by the model.")

            # Extract topics and format them for display
            topics = []
            for topic_id, topic_words in lda_model.show_topics(formatted=False, num_words=10):
                topics.append({
                    "Topic": topic_id,
                    "Keywords": ", ".join([word for word, _ in topic_words])
                })
            df_topics = pd.DataFrame(topics)
            st.dataframe(df_topics, use_container_width=True)

        with sub_tab_orgs:
            st.subheader("Topic Distribution by Organization")
            st.markdown("Which topics are most frequently used by which organizations (based on your filters)? Please "
                        "note that the graph is not easy to read unless you enter fullscreen mode.")

            # Use the 'dominant_topic' column we merged earlier
            fig_topic_orgs = px.histogram(
                filtered_df,
                x='Name',
                color='dominant_topic_name',  # Use the topic names here
                barmode='stack',
                title="Topic Distribution for Top Organizations in Filter",
                labels={'Name': 'Organization', 'dominant_topic_name': 'Topic Name'}  # Update the label
            )
            fig_topic_orgs.update_xaxes(categoryorder='total descending')
            st.plotly_chart(fig_topic_orgs, use_container_width=True)
    elif analysis_type == 'Geographical':
        coords_df = filtered_df[['Delivered_Date', 'Longitude', 'Latitude']]
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=coords_df,
            get_position='[Longitude, Latitude]',
            get_fill_color='[255, 255, 0, 160]',  # Yellow with some transparency
            get_radius=200,
            pickable=True,
        )

        # Define the map view
        view_state = pdk.ViewState(
            latitude=coords_df['Latitude'].mean(),
            longitude=coords_df['Longitude'].mean(),
            zoom=4,
            pitch=0,
        )

        # Render the deck.gl map
        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state
        ))
except ValueError:
    st.write('You have only selected one date! Please select another in order to filter the data.')