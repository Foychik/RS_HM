import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from funcs import *
import streamlit.components.v1 as components

def main():

    st.set_page_config(layout="wide", initial_sidebar_state='expanded')
        
    sidebar_header = '''This is a demo to illustrate a recommender system that finds similar items to a given clothing article or recommend items for a customer using 4 different approaches:'''
    
    page_options = ["Найти похожие вещи",
                    "Рекомендации покупателю",
                    'Документация']
    
    st.sidebar.info(sidebar_header)


    
    page_selection = st.sidebar.radio("Try", page_options)
    articles_df = pd.read_csv('articles.csv')
    
    models = ['Похожие товары на основе эмбеддингов картинок',
              'Похожие товары на основе эмбеддингов текстового описания',
              'Похожие товары на основе эмбеддингов табличных признаков (категория, цвет и т.д.)',
              'Похожие товары на основе эмбеддингов TensorFlow Recommendrs модели',
              'Похожие товары на основе комбинации эмбеддингов']
    
    model_descs = ['Эмбеддинги картинок посчитаны с помощью VGG16 CNN с Keras',
                  'Эмбеддинги текстового описания получены с помощью "universal-sentence-encoder" из TensorFlow Hub',
                  'Эмбеддинги табличных фичей посчитаны с помощью one-hot encoding',
                  'TFRS нейросетевая модель позволяет сделать коллаборативную фильтрацию и отранжировать',
                  'Конкатенация всех эмбеддингов позволяет более точно найти похожие объекты']

#########################################################################################
#########################################################################################

    if page_selection == "Найти похожие вещи":

        articles_rcmnds = pd.read_csv('results/articles_rcmnds.csv')

        articles = articles_rcmnds.article_id.unique()
        get_item = st.sidebar.button('Получить случайную вещь')
        
        if get_item:
            
            rand_article = np.random.choice(articles)
            article_data = articles_rcmnds[articles_rcmnds.article_id == rand_article]
            rand_article_desc = articles_df[articles_df.article_id == rand_article].detail_desc.iloc[0]
            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(article_data)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            scores = get_rcmnds_scores(article_data)
            features = get_rcmnds_features(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            images = get_rcmnds_images(image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            st.sidebar.image(get_item_image(str(rand_article), width=200, height=300))
            st.sidebar.write('Описание')
            st.sidebar.caption(rand_article_desc)

            with st.container():     
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded = model == 'Похожие товары на основе эмбеддингов картинок' or model == 'Похожие товары на основе эмбеддингов текстового описания')
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_container_width=True)
                                if model == 'Похожие товары на основе эмбеддингов текстового описания':
                                    st.caption(detail_desc)
                                    
#########################################################################################
#########################################################################################

    if page_selection == "Рекомендации покупателю":
        
        customers_rcmnds = pd.read_csv('results/customers_rcmnds.csv')
        customers = customers_rcmnds.customer.unique()        
        
        get_item = st.sidebar.button('Показать случайного покупателя')
        if get_item:
            st.sidebar.write('#### История покупок')

            rand_customer = np.random.choice(customers)
            customer_data = customers_rcmnds[customers_rcmnds.customer == rand_customer]
            customer_history = np.array(eval(customer_data.history.iloc[0]))

            image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds = get_rcmnds(customer_data)
            
            scores = get_rcmnds_scores(customer_data)
            features = get_rcmnds_features(articles_df, combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            images = get_rcmnds_images(combined_rcmnds, tfrs_rcmnds, image_rcmnds, text_rcmnds, feature_rcmnds)
            detail_descs  = get_rcmnds_desc(articles_df, image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)
            
            rcmnds = (image_rcmnds, text_rcmnds, feature_rcmnds, tfrs_rcmnds, combined_rcmnds)

            splits = [customer_history[i:i+3] for i in range(0, len(customer_history), 3)]
                            
            for split in splits:
                with st.sidebar.container():
                    cols = st.columns(3)
                    for item, col in zip(split, cols):
                        col.image(get_item_image(str(item), 100))
                    

            with st.container():            
                for i, model, image_set, score_set, model_desc, detail_desc_set, features_set, rcmnds_set in zip(range(5), models, images, scores, model_descs, detail_descs, features, rcmnds):
                    container = st.expander(model, expanded=True)
                    with container:
                        cols = st.columns(7)
                        cols[0].write('###### Similarity Score')
                        cols[0].caption(model_desc)
                        for img, col, score, detail_desc, rcmnd in zip(image_set[1:], cols[1:], score_set[1:], detail_desc_set[1:],  rcmnds_set[1:]):
                            with col:
                                st.caption('{}'.format(score))
                                st.image(img, use_container_width=True)
                                

#########################################################################################  
#########################################################################################

    if page_selection == "Документация":
                
        components.html(
            """
           <header style="color: white;">
           

        <h2>Рекомендательная система по подбору одежды</h2>
        
        Датасет взят из соревнования на kaggle, которое проводилось H&M
        <a href="https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations", target="_blank">Competition Page</a>.
        
        <br>
        В проекте использованы 2 подхода решения задачи рекомендаций:
        <ul>
        <li>Фильтрация на основе контента</li>
        <li>Коллаборатоивная фильтрация</li>
        </ul>
        
        <br>
        Идея реализации систем рекомендаций заключается в построении представлений клиентов и товаров. Эти представления называются эмбеддингами, которые являются проекцией клиентов и товаров в N-мерное пространство.
        <br>
        <br>
        Фильтрация на основе контента использует характеристики товаров для рекомендации других товаров, похожих на те, которые нравятся пользователю, на основе его предыдущих действий или явной обратной связи. Эти характеристики используются для построения эмбеддингов товаров. В этом проекте мы использовали характеристики, предоставленные H&M, которые описывают каждый товар, такие как цвет, текстура, тип одежды и т. д.
        <br>
        <br>
        Кроме того, мы построили ещё 2 набора эмбеддингов: на основе картинок и на основе текстовых описаний. Эмбеддинги картинок мы получили с помощью предобученной VGG16 (сверточная нейронка).
        Для получения текстовых эмбеддингов мы использовали предобученный универсальный "universal-sentence-encoder".
        
        <br>
        <br>
        Коллаборативная фильтрация использует матрицу взаимодействия товаров и пользователей, чтобы устранить некоторые ограничения фильтрации на основе контента, а именно необходимость инжиниринга характеристик товаров. Матрица взаимодействия затем факторизуется на два набора N-мерных эмбеддингов, один для пользователей и один для товаров.
        <br>
        
        <h2>Список подготовительных ноутбуков в проекте:</h2>
        <ul>
        
        <li>Схожесть товаров по текстовым описаниям

        <a target="_blank">
       product-similarity-with-text-embeddings</a> 
        
        </li>
        <li>Эмбеддинги покупателя и товара из текста

        <a target="_blank">
        h-m-article-and-customer-embeddings-from-text-desc</a> 
        
        </li>
        <br>
        <li>
        Схожесть продуктов по картинкам

        <a target="_blank">
        finding-similar-items-with-image-embeddings-knn</a> 

        </li>
        <li>Эмбеддинги товаров по картинкам

        <a target="_blank">
        product-embeddings-from-images-keras-vgg16</a> 
        </li>
        
        <li>Эмбеддинги покупателей по картинкам

        <a target="_blank">
        hm-product-image-embeddings</a> 
        </li>
        <br>
        <li>
        Схожесть продуктов по табличным характеристикам

        <a  target="_blank">
        content-based-filtering-with-pca</a> 

        </li>
        <li>Эмбеддинги покупателей по табличным характеристикам
        <a target="_blank">
        customer-embeddings-from-features</a> 
        </li>
        
        <li>Эмбеддинги товаров по табличным признакам
        <a target="_blank">
        article-embeddings-from-features</a> 
        
        </li>
        <br>
        <li>
        Схожесть товаров с TFRS (коллаборативная фильтрация)
        <a target="_blank">
        h-m-basic-retrieval-model-tf-recommender</a> 
        <br>
        </li>
        <br>
        <li>
        
        Сравнение 4 подходов

        <a target="_blank">
        comparing-4-different-approaches</a> 
        </li>
        
        <br>
        
        </ul>
        </header>
        
        
        
            """,
            height=1000,
)

if __name__ == '__main__':
    main()
