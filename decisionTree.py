import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib as plt
iris = load_iris()
# print (iris.feature_names)
# print (iris.target_names)
# print (iris.data[0])
# print (iris.target[0])
test_idx = [0, 50, 100]

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)
tree.export_graphviz(clf,out_file='tree.dot')



# print(test_target)
# print(clf.predict(test_data))

from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

print(graph)  # [<pydot.Dot object at 0x000001F7BD1A9630>]
#print(graph[0])  # <pydot.Dot object at 0x000001F7BD1A9630>
graph.write_pdf("iris.pdf")
#graph[0].write_pdf("iris.pdf")