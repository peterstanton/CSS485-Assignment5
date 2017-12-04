clear

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
Input = loadMNISTImages('train-images.idx3-ubyte'); %inputs
Output = loadMNISTLabels('train-labels.idx1-ubyte'); %outputs

t0 = [1 0 0 0 0 0 0 0 0 0];
t1 = [0 1 0 0 0 0 0 0 0 0];
t2 = [0 0 1 0 0 0 0 0 0 0];
t3 = [0 0 0 1 0 0 0 0 0 0];
t4 = [0 0 0 0 1 0 0 0 0 0];
t5 = [0 0 0 0 0 1 0 0 0 0];
t6 = [0 0 0 0 0 0 1 0 0 0];
t7 = [0 0 0 0 0 0 0 1 0 0];
t8 = [0 0 0 0 0 0 0 0 1 0];
t9 = [0 0 0 0 0 0 0 0 0 1];
 
answers = [t0; t1; t2; t3; t4; t5; t6; t7; t8; t9;];
% We are using display_network from the autoencoder code
% display_network(images(:,1:100)); % Show the first 100 images
 LearningRate = 0.001;
  
NumHidLayerNeurons = 70;
NumOutLayer = 10;

% r = (b-a).*rand(1000,1) + a;

 
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
%     if i == 6000
%         i = 1;
%     else
%         i = i + 1;
%     end
    i = floor((rand(1)*6001) + 1);
    IterationCount = IterationCount + 1           %Increment the counter
    outOfHidden = softmax(HiddenLayerWeights * Input(:,i) + biasHidden);   
    outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
    
    targetNumber = Output(i,:);
    targetAnswer = answers(targetNumber+1,:)';
    
  % e = t(:,(num+1)) - outputResult;
   
   myError = targetAnswer - outOfOutput;
   
%    s2 = -2.*diag((ones(size(outputResult))-outputResult).*outputResult)*e;
%    s1 = diag((ones(size(hiddenResult))-hiddenResult).*hiddenResult)*outputWeights'*s2;

  
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
