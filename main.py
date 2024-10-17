import numpy as np
import tensorflow as tf
from tensorflow import keras
from recsys_utils import * 

def cost_func(X, W, b, Y, R, lambda_):
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2)) 
    return J

X, W, b, num_movies, num_features, num_users = load_precalc_params_small()
Y, R = load_ratings_small()

movieList, movieList_df = load_Movie_List_pd()

# Khởi tạo các rating ban đầu của user
my_ratings = np.zeros(num_movies)         

n = int(input("Nhập số lượng phim bạn muốn đánh giá: "))
for i in range(n):
    idx = int(input("Nhập mã phim mà bạn muốn đánh giá: "))
    rate = float(input("Đánh giá của bạn (1 - 5): "))
    my_ratings[idx] = rate

# Tạo ra mảng R tương ứng với user mới
my_R = np.array(my_ratings != 0).astype(int)

Y = np.c_[my_ratings, Y]
R = np.c_[my_R, R]

Ynorm, Ymean = normalizeRatings(Y, R)
num_movies, num_users = Y.shape

tf.random.set_seed(1234) 
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Cho learning rate bắt đầu ở 0,1
optimizer = keras.optimizers.Adam(learning_rate=1e-1)

iterations = 200
lambda_ = 1

for iter in range(iterations):

    with tf.GradientTape() as tape:

        cost_value = cost_func(X, W, b, Ynorm, R, lambda_)

    grads = tape.gradient(cost_value,[X,W,b])

    optimizer.apply_gradients(zip(grads,[X,W,b]))

p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()


pm = p + Ymean

my_predictions = pm[:,0]

print("Dự đoán rating dành cho các bộ phim khác trong data:")
for i in range(len(my_predictions)):
    print(f'Dự đoán {my_predictions[i]:.2f} dành cho {movieList[i]}')


print('\nĐánh giá ban đầu và đánh giá được dự đoán thông qua chương trình:\n')
for i in range(len(my_ratings)):
    if my_ratings[i] > 0:
        print(f'Đánh giá ban đầu: {my_ratings[i]}, Dự đoán {my_predictions[i]:0.2f} dành cho {movieList[i]}')
