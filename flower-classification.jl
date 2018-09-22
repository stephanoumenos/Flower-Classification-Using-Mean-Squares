# Author: Stefano De Checchi
# This program trains and tests mean square classifiers to determine the flower type
# Subject Professor: Matheus Souza 
# To exec just run:
# $ julia flower-classification.jl

#########################################
# Functions
#########################################

function test_independent_classifier(classifier_flower, classifier_output, answer)
    answer_length = length(answer)
    corrects = 0
    wrong = 0
    if length(classifier_output) != length(answer)
        println("Error: lenghts must be equal")
        exit(1)
    end
    for i = 1:answer_length
        tested_element = classifier_output[i]
        if tested_element != 1 && answer[i] == classifier_flower
            wrong += 1
        else
            corrects += 1
        end
    end
    error_rate = wrong/(wrong+corrects)
    return error_rate
end


function get_flower_types(outputs)
    flower_types = []
    samples = length(outputs[1])
    for i = 1:samples
        flower_type = max(outputs[1][i], outputs[2][i], outputs[3][i])
        if flower_type == outputs[1][i]
            push!(flower_types, 1)
        elseif flower_type == outputs[2][i]
            push!(flower_types, 2)
        else
            push!(flower_types, 3)
        end
    end
    return flower_types
end


function get_confusion_matrix(flower_types, answer)
    if length(flower_types) != length(answer)
        println("Error: lenghts must be equal")
        exit(1)
    end
    number_of_flowers = 3
    confusion_matrix = zeros(number_of_flowers, number_of_flowers)
    for i = 1:length(answer)
        confusion_matrix[Int(answer[i]), Int(flower_types[i])] += 1 
    end
    return confusion_matrix
end


function print_confusion_matrix(confusion_matrix)
    println()
    println("              Prediction")
    println("              --------------")
    println("Saída (y)     1    2   3")
    println("----------------------------")
    println("  1          ", confusion_matrix[1,1], "   ", confusion_matrix[1,2], "   ", confusion_matrix[1,3])
    println("  2          ", confusion_matrix[2,1], "    ", confusion_matrix[2,2], "   ", confusion_matrix[2,3])
    println("  3          ", confusion_matrix[3,1], "    ", confusion_matrix[3,2], "   ", confusion_matrix[3,3])
    println()
end


#########################################
# Main Routine
#########################################

# Atributes for each flower in the X matrix
atributes = ["sepal length (cm)",
             "sepal width  (cm)",
             "petal length (cm)",
             "petal width  (cm)"]

# y = 1 means it's a setosa flower, y = 2 it's a versicolored and so on...
rotulos = ["setosa", "versicolored", "virginic"]

X = [4.9 4.8 6.0 6.4 5.1 6.7 4.3 6.8 6.3 6.7 4.9 4.4 4.6 7.9 5.8 7.0 7.7 5.6 5.7 5.9 5.0 5.7 6.4 5.8 5.1 6.7 6.0 6.4 6.3 4.8 5.5 4.7 5.5 5.0 6.3 6.4 6.3 4.9 6.5 5.7 6.6 6.0 6.0 5.5 6.7 5.6 6.7 6.1 5.1 7.7 7.2 5.5 5.2 6.5 5.8 7.2 5.4 6.3 7.6 6.4 6.7 6.5 5.6 5.7 4.7 5.0 5.4 5.0 6.3 5.5 6.1 6.2 6.1 5.3 6.3 6.2 5.6 5.0 5.8 5.8 6.2 5.4 6.5 5.9 4.9 4.6 7.2 6.9 6.4 5.5 6.0 6.3 4.9 4.8 6.7 5.1 7.3 5.1 6.9 4.9 6.1 7.7 6.1 5.9 4.6 5.2 6.9 5.8 6.3 6.1 5.4 4.8 7.7 6.7 4.5 5.4 5.0 5.1 5.6 4.6 5.7 5.0 5.7 6.8 6.4 5.0 5.7 5.6 5.0 4.8 5.2 7.1 6.5 4.4 6.2 6.8 7.4 5.1 5.1 5.5 4.4 6.0 6.6 5.1 5.8 5.4 6.9 5.0 5.7 5.2; 
    3.1 3.4 2.7 3.2 3.8 3.1 3.0 3.2 2.8 3.1 3.6 3.2 3.6 3.8 2.6 3.2 2.6 3.0 2.8 3.2 3.2 3.8 2.8 2.7 3.7 3.0 3.0 2.8 2.5 3.0 2.6 3.2 4.2 3.5 2.9 2.9 2.5 2.5 2.8 2.6 2.9 2.2 2.2 2.3 2.5 2.7 3.0 2.6 2.5 2.8 3.6 2.5 2.7 3.0 2.7 3.0 3.4 2.7 3.0 3.1 3.3 3.2 2.9 2.5 3.2 3.3 3.0 3.0 3.4 3.5 2.9 2.8 3.0 3.7 3.3 2.2 2.5 3.4 2.7 4.0 2.9 3.7 3.0 3.0 3.0 3.2 3.2 3.1 3.2 2.4 2.9 2.3 3.1 3.0 3.1 3.8 2.9 3.4 3.1 2.4 2.8 3.0 3.0 3.0 3.4 3.5 3.2 2.8 3.3 2.8 3.9 3.1 3.8 3.3 2.3 3.9 3.6 3.5 3.0 3.1 2.9 3.4 4.4 2.8 2.7 2.0 2.8 2.8 2.3 3.4 4.1 3.0 3.0 3.0 3.4 3.0 2.8 3.3 3.5 2.4 2.9 3.4 3.0 3.8 2.7 3.4 3.1 3.5 3.0 3.4; 
    1.5 1.6 5.1 4.5 1.9 4.7 1.1 5.9 5.1 5.6 1.4 1.3 1.0 6.4 4.0 4.7 6.9 4.5 4.1 4.8 1.2 1.7 5.6 4.1 1.5 5.2 4.8 5.6 5.0 1.4 4.4 1.3 1.4 1.3 5.6 4.3 4.9 4.5 4.6 3.5 4.6 4.0 5.0 4.0 5.8 4.2 5.0 5.6 3.0 6.7 6.1 4.0 3.9 5.2 5.1 5.8 1.5 4.9 6.6 5.5 5.7 5.1 3.6 5.0 1.6 1.4 4.5 1.6 5.6 1.3 4.7 4.8 4.9 1.5 4.7 4.5 3.9 1.5 3.9 1.2 4.3 1.5 5.8 5.1 1.4 1.4 6.0 5.4 5.3 3.8 4.5 4.4 1.5 1.4 4.4 1.6 6.3 1.5 5.1 3.3 4.7 6.1 4.6 4.2 1.4 1.5 5.7 5.1 6.0 4.0 1.3 1.6 6.7 5.7 1.3 1.7 1.4 1.4 4.1 1.5 4.2 1.6 1.5 4.8 5.3 3.5 4.5 4.9 3.3 1.9 1.5 5.9 5.5 1.3 5.4 5.5 6.1 1.7 1.4 3.7 1.4 4.5 4.4 1.5 5.1 1.7 4.9 1.6 4.2 1.4; 
    0.1 0.2 1.6 1.5 0.4 1.5 0.1 2.3 1.5 2.4 0.1 0.2 0.2 2.0 1.2 1.4 2.3 1.5 1.3 1.8 0.2 0.3 2.1 1.0 0.4 2.3 1.8 2.2 1.9 0.3 1.2 0.2 0.2 0.3 1.8 1.3 1.5 1.7 1.5 1.0 1.3 1.0 1.5 1.3 1.8 1.3 1.7 1.4 1.1 2.0 2.5 1.3 1.4 2.0 1.9 1.6 0.4 1.8 2.1 1.8 2.5 2.0 1.3 2.0 0.2 0.2 1.5 0.2 2.4 0.2 1.4 1.8 1.8 0.2 1.6 1.5 1.1 0.2 1.2 0.2 1.3 0.2 2.2 1.8 0.2 0.2 1.8 2.1 2.3 1.1 1.5 1.3 0.2 0.1 1.4 0.2 1.8 0.2 2.3 1.0 1.2 2.3 1.4 1.5 0.3 0.2 2.3 2.4 2.5 1.3 0.4 0.2 2.2 2.1 0.3 0.4 0.2 0.3 1.3 0.2 1.3 0.4 0.4 1.4 1.9 1.0 1.3 2.0 1.0 0.2 0.1 2.1 1.8 0.2 2.3 2.1 1.9 0.5 0.2 1.0 0.2 1.6 1.4 0.3 1.9 0.2 1.5 0.6 1.2 0.2]';

y  = [1.0,1.0,2.0,2.0,1.0,2.0,1.0,3.0,3.0,3.0,1.0,1.0,1.0,3.0,2.0,2.0,3.0,2.0,2.0,2.0,1.0,1.0,3.0,2.0,1.0,3.0,3.0,3.0,3.0,1.0,2.0,1.0,1.0,1.0,3.0,2.0,2.0,3.0,2.0,2.0,2.0,2.0,3.0,2.0,3.0,2.0,2.0,3.0,2.0,3.0,3.0,2.0,2.0,3.0,3.0,3.0,1.0,3.0,3.0,3.0,3.0,3.0,2.0,3.0,1.0,1.0,2.0,1.0,3.0,1.0,2.0,3.0,3.0,1.0,2.0,2.0,2.0,1.0,2.0,1.0,2.0,1.0,3.0,3.0,1.0,1.0,3.0,3.0,3.0,2.0,2.0,2.0,1.0,1.0,2.0,1.0,3.0,1.0,3.0,2.0,2.0,3.0,2.0,2.0,1.0,1.0,3.0,3.0,3.0,2.0,1.0,1.0,3.0,3.0,1.0,1.0,1.0,1.0,2.0,1.0,2.0,1.0,1.0,2.0,3.0,2.0,2.0,3.0,2.0,1.0,1.0,3.0,3.0,1.0,3.0,3.0,3.0,1.0,1.0,2.0,1.0,2.0,2.0,1.0,3.0,1.0,2.0,1.0,2.0,1.0]';


#########################################
# Calculating Classifiers
#########################################
classifiers = []
n = length(y)
training_samples = Int(n*2/3)
testing_samples = Int(n*1/3)
println("Used atributes")
for i in 1:4
    println(i, "/4: ", atributes[i])
end
println("-------- Calculating Classifiers --------")
print("Training samples:")
println(training_samples)
print("Testing samples:")
println(testing_samples)

X_train = X[1:training_samples, 1:4]
y_train = y[1:training_samples]
X_test = X[training_samples+1:training_samples+testing_samples, 1:4]
y_test = y[training_samples+1:training_samples+testing_samples]

for FLOWER_TYPE in 1:3
    print("Flower ", FLOWER_TYPE, " of 3 (", rotulos[FLOWER_TYPE], ")...")
    A = X_train

    b = []
    for y_i in y_train
        if y_i == FLOWER_TYPE
            push!(b, 1)
        else
            push!(b,-1)
        end
    end

    classifier = inv(X_train'*X_train) * X_train' * b
    push!(classifiers, classifier)
    println("Done")
end

#########################################
# Testing Classifiers
#########################################

println("-------- Testing                 --------")

outputs_training = [X_train * classifier for classifier in classifiers]
outputs_test = [X_test * classifier for classifier in classifiers]

# Independent Testing
println("Step 1/2: Independent Testing...")
for i = 1:3
    println("Classifier ", i, ": ---", rotulos[i], "---")
    error_rate_training = test_independent_classifier(i,
                                                      outputs_training[i],
                                                      y_train)
    println("Error rate training set: ", error_rate_training*100, "%")
    error_rate_testing = test_independent_classifier(i,
                                                     outputs_test[i],
                                                     y_test)
    println("Error rate testing set: ", error_rate_testing*100, "%")
end


# 3-Class Classifier Testing
println("Step 2/2: 3 Class Classifier Testing...")

# This is the final result containing the flower type
flower_types_training = get_flower_types(outputs_training)
flower_types_test = get_flower_types(outputs_test)

confusion_matrix_training = get_confusion_matrix(flower_types_training,
                                                 y_train)
confusion_matrix_test = get_confusion_matrix(flower_types_test,
                                                y_test)
println("\nConfusion Matrix Training Set:\n")

print_confusion_matrix(confusion_matrix_training)

println("\nConfusion Matrix Testing Set:\n")

print_confusion_matrix(confusion_matrix_test)
