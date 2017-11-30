% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
Input = loadMNISTImages('train-images.idx3-ubyte')'; %inputs
Output = loadMNISTLabels('train-labels.idx1-ubyte')'; %outputs
 
% We are using display_network from the autoencoder code
% display_network(images(:,1:100)); % Show the first 100 images
disp(labels(1:10));

LearningRate = 0.05;

NumHidLayerNeurons = 500;

HiddenLayerWeights = rand(NumHidLayerNeurons,784); % Weight matrix from Input to Hidden
OutputLayerWeights = rand(15,NumHidLayerNeurons); % Weight matrix from Hidden to Output
biasHidden = rand(NumHidLayerNeurons,1);         % Random bias of the hidden layer
biasOutput = rand(15,1);  %Bias of the output layer, DONT KNOW SIZE OF OUTPUT YET

IterationCount = 0;

ErrorVec1(1:50000) = 0;
xVec(1:50000) = 0;
tester = 10.00;
i = 0;

while(IterationCount < 50000)
    if i == 6000
        i = 1;
    else
        i = i + 1;
    end
    IterationCount = IterationCount + 1;           %Increment the counter
    outOfHidden = logsig(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
   myError = Output(:,i) - outOfOutput;
  
   S2 = -2.*diag(ones(size(outOfOutput))-outOfOutput.*outOfOutput)*myError;   
   S1 = diag(ones(size(outOfHidden))-outOfHidden.*outOfHidden)*OutputLayerWeights'*S2;
         
   OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
   HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';
   
   biasOutput = biasOutput - LearningRate.*S2;
   biasHidden = biasHidden - LearningRate.*S1;
   
   xVec(IterationCount) = IterationCount;
   ErrorVec1(IterationCount) = sum(myError.^2)/length(myError);
   tester = sum(myError.^2);    
end