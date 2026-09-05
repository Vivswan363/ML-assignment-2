import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt 

eps = 1e-8

def normalize(df):
    columns = [col for col in df.columns if (col != 'label' and col != 'release_id')]
    train_stats = {col: [] for col in columns}
    for col in columns:
        col_mean = df[col].mean()
        col_std = df[col].std()
        train_stats[col] = [col_mean, col_std]
        df[col] = (df[col] - col_mean) / (col_std + eps)
    return train_stats, df

def fit_normalize(train_stats, df):
    for col in train_stats:
        col_mean = train_stats[col][0]
        col_std = train_stats[col][1]
        df[col] = (df[col] - col_mean) / (col_std + eps)
    return df

def load_data(path):
    df = pd.read_csv(path)
    return df

def prepare_train_data(train_df):
    columns = [col for col in train_df.columns if (col != 'label' and col != 'release_id')]
    X_train = train_df[columns].values
    y_train = train_df['label'].values.reshape(-1, 1)
    y_train = OneHotEncoder(sparse_output=False).fit_transform(y_train)
    return X_train, y_train

def gradient_descent(X, y, rng, batch_size, lr=0.3, epochs=500, ada=False, 
                     class_weighing=False, class_weighing_power=1):
    W = np.zeros((X.shape[1], y.shape[1]))
    b = np.zeros(y.shape[1])
    
    num_batches = X.shape[0] // batch_size 
    
    n = X.shape[0]
    
    train_losses = []
    
    accum_w = np.zeros_like(W) 
    accum_b = np.zeros_like(b) 
    
    n_i = np.sum(y, axis=0).flatten()
    n_data = sum(n_i) 
    
    alpha_i = n_data / (3 * n_i) 
    
    for epoch in range(epochs):
        print(f"epoch: {epoch + 1}/{epochs}")
        batch_losses = np.zeros(num_batches + 1)
        
        order = rng.permutation(n)
        
        for i in range(num_batches + 1):
            
            if i == num_batches and batch_size * num_batches == X.shape[0]:
                continue
            
            indices = order[i * batch_size: (i + 1) * batch_size]
            X_B = X[indices]
            y_B = y[indices]
            alpha_B = np.sum(y_B * alpha_i, axis=1, keepdims=True)
            
        
            logits = X_B @ W + b
            logits = logits - np.max(logits, axis=1, keepdims=True) 
            
            P_B = np.exp(logits) 
            P_B = P_B / np.sum(P_B, axis=1, keepdims=True) 
            
            loss = -np.mean(np.log(np.sum(y_B * P_B, axis=1)))
            
            batch_losses[i] = loss
            
            confusion = P_B - y_B
            
            if class_weighing: 
                alpha_B = np.power(alpha_B, class_weighing_power)
                confusion *= alpha_B 
                confusion /= np.sum(alpha_B.flatten())
            
            g_w = X_B.T @ confusion / batch_size 
            g_b = np.sum(confusion, axis=0) / batch_size
            
            # print(g_w, g_b)
            # exit(1)
            
            if ada: 
                accum_w += g_w * g_w 
                accum_b += g_b * g_b 
                g_w /= np.sqrt(accum_w) + eps
                g_b /= np.sqrt(accum_b) + eps
             
            W -= lr * g_w 
            b -= lr * g_b 
            
        
        mean_loss = batch_losses.mean() if batch_losses[-1] != 0 else batch_losses[:-1].mean()
        train_losses.append(mean_loss)
    return train_losses, W, b 

def predict_prob(X, W, b):
    logits = X @ W + b 
    prob = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    prob /= np.sum(prob, axis=1, keepdims=True)
    return prob

def model_performance(y_true, y_model):
    aps = average_precision_score(y_true, y_model)
    print(f"average precision score = {aps}")
    return 

def plot_loss(losses, path):
    plt.plot(losses) 
    plt.ylabel("loss values")
    plt.xlabel("steps")
    plt.title("loss vs time")
    plt.savefig(f"{path}.png")

if __name__ == "__main__":
    
    rng = np.random.default_rng(seed=774)
    
    train_df = load_data("data/part_ab_train.csv")
    
    train_stats, df = normalize(train_df)
    
    X_train, y_train = prepare_train_data(df) 
    
    losses, W, bias = gradient_descent(X_train, y_train, rng, epochs=200, batch_size=32, lr=0.3, ada=1, 
                                       class_weighing=True, class_weighing_power=1)

    
    plot_loss(losses, "class_weighing_ada_power")
    
    val_df = load_data("data/part_ab_val.csv")
    val_df = fit_normalize(train_stats, val_df)
    
    X_val, y_val = prepare_train_data(val_df) 
    y_model_val = predict_prob(X_val, W, bias) 
    # print(y_model_val)
    # exit(1)
    
    model_performance(y_true=y_val, y_model=y_model_val)
    
    
    
    
        
        
            
            
            
            