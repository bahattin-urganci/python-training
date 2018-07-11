#from sklearn import tree
#features=[[140,1],[130,1],[150,0],[170,0]]
#labels=[0,0,1,1]
#clf=tree.DecisionTreeClassifier()
#clf=clf.fit(features,labels)
#print(clf.predict([[150,0]]))

import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

