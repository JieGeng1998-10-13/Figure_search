from sklearn.cluster import KMeans
import pickle
import joblib

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)
    
X = [item['embedding'] for item in embeddings]
kmeans = KMeans(n_clusters=50)
kmeans.fit(X)
preds = kmeans.predict(X)


for item, pred in zip(embeddings, preds):
    item['cluster'] = pred
joblib.dump(kmeans, 'kmeans.pkl')
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

