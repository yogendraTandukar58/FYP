import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from orders.models import OrderProduct
from store.models import Product, ReviewRating
from django.db.models import Avg
def generate_model():
    # Fetch data from Django models
    products = Product.objects.all()
    products_df= pd.DataFrame(list(products))
    user_history = OrderProduct.objects.all()
    user_history_df=pd.DataFrame(list(user_history))
    print(products_df)
    print(user_history_df)
    average_rating = ReviewRating.objects.values('product_id').annotate(average_rating=Avg('rating'))
    average_rating_df = pd.DataFrame(list(average_rating))

    # Merge user history with product information
    user_history_with_product = pd.merge(user_history_df, products_df, on='product_id', how='left')
    user_history_with_product = pd.merge(user_history_with_product, average_rating_df, on='product_id', how='left')

    # Create user-item matrix
    user_item_matrix = pd.pivot_table(user_history_with_product, index='user_id', columns='product_id', values='average_rating', fill_value=0)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(user_item_matrix.T)
    cosine_sim

    # Implement KNN
    knn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    knn.fit(cosine_sim)

    # Generate recommendations
    user_id = 1  # Replace with the user ID for whom you want recommendations
    user_history_for_user = user_item_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = knn.kneighbors(user_history_for_user)

    user_history_for_user

    # Get recommended product IDs
    recommended_product_ids = user_item_matrix.columns[indices.flatten()]

    # Display recommendations with product information
    recommended_products = products[products['product_id'].isin(recommended_product_ids)]
    print(recommended_products[['product_title', 'category']])

    import joblib
    # Save the model to a file
    joblib.dump(knn, 'assets/knn_model_lite2.joblib')
