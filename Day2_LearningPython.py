#Training and Testing
X = df.drop('Purchased',axis=1)
y = df['Purchased']

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

print("Training Feature:\n",X_train)
print("----------------------------------------------------------------")
print("Testing Feature:\n",X_test)
print("----------------------------------------------------------------")
print("Training Label:\n",y_train)
print("----------------------------------------------------------------")
print("Testing Label:\n",y_test)
print("----------------------------------------------------------------")

