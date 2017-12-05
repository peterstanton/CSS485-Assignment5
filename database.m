% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
Input = loadMNISTImages('train-images.idx3-ubyte'); %inputs
Output = loadMNISTLabels('train-labels.idx1-ubyte'); %outputs
 
% We are using display_network from the autoencoder code
% display_network(images(:,1:100)); % Show the first 100 images
 LearningRate = 0.001;
  
NumHidLayerNeurons = 200;
NumOutLayer = 4;
 
HiddenLayerWeights = rand(NumHidLayerNeurons,784); % Weight matrix from Input to Hidden
OutputLayerWeights = rand(NumOutLayer,NumHidLayerNeurons); % Weight matrix from Hidden to Output
biasHidden = rand(NumHidLayerNeurons,1);         % Random bias.
biasOutput = rand(NumOutLayer,1);

tarIterations = 10000;
IterationCount = 0;  %Count passes.
ErrorVec1(1:tarIterations) = 0;
xVec(1:tarIterations) = 0;
i = 0;
while(IterationCount < tarIterations)
    if i == 6000
        i = 1;
    else
        i = i + 1;
    end
    IterationCount = IterationCount + 1           %Increment the counter
    outOfHidden = softmax(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
    targetNumber = Output(i,:);
    targetAnswer = decimalToBinaryVector(targetNumber,NumOutLayer)';

    
   myError = targetAnswer - outOfOutput;
  
   S2 = -2.*diag((ones(size(outOfOutput))-outOfOutput).*outOfOutput)*myError;   
   S1 = diag((ones(size(outOfHidden))-outOfHidden).*outOfHidden)*OutputLayerWeights'*S2;

   OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
   HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';
   
   biasOutput = biasOutput - LearningRate.*S2;
   biasHidden = biasHidden - LearningRate.*S1;
   
   xVec(IterationCount) = IterationCount;
   ErrorVec1(IterationCount) = sum(myError.^2)/length(myError);
end

figure(1)
plot(xVec,ErrorVec1)
title('Backpropagation Network Training')
xlabel('Backpropagation Iterations')
ylabel('Squared Error')