clear

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
Input = loadMNISTImages('train-images.idx3-ubyte'); %inputs
Output = loadMNISTLabels('train-labels.idx1-ubyte'); %outputs

t0 = [1,0,0,0,0,0,0,0,0,0];
t1 = [0,1,0,0,0,0,0,0,0,0];
t2 = [0,0,1,0,0,0,0,0,0,0];
t3 = [0,0,0,1,0,0,0,0,0,0];
t4 = [0,0,0,0,1,0,0,0,0,0];
t5 = [0,0,0,0,0,1,0,0,0,0];
t6 = [0,0,0,0,0,0,1,0,0,0];
t7 = [0,0,0,0,0,0,0,1,0,0];
t8 = [0,0,0,0,0,0,0,0,1,0];
t9 = [0,0,0,0,0,0,0,0,0,1];
 
answers = [t0; t1; t2; t3; t4; t5; t6; t7; t8; t9;];
% We are using display_network from the autoencoder code
% display_network(images(:,1:100)); % Show the first 100 images
 LearningRate = 0.001;
  
NumHidLayerNeurons = 75;
NumOutLayer = 10;

a = -0.01;
b = 0.01;
% r = (b-a).*rand(1000,1) + a;
 
HiddenLayerWeights = (b-a).*rand(NumHidLayerNeurons,784) + a; % Weight matrix from Input to Hidden
OutputLayerWeights = (b-a).*rand(NumOutLayer,NumHidLayerNeurons) + a; % Weight matrix from Hidden to Output
biasHidden = (b-a).*rand(NumHidLayerNeurons,1) + a;         % Random bias.
biasOutput = (b-a).*rand(NumOutLayer,1) + a;

tarIterations = 10000;

Epochs = 10;

ErrorVecMaster(Epochs:tarIterations) = 0;

IterationCount = 0;  %Count passes.
% ErrorVec1(1:tarIterations) = 0;
 xVec(1:tarIterations) = 0;
i = 0;
thisEpoch = 1;
for e = 1:Epochs
    while(IterationCount < tarIterations)
        i = floor((rand(1)*6001) + 1);
        IterationCount = IterationCount + 1         %Increment the counter
        outOfHidden = tanh(HiddenLayerWeights * Input(:,i) + biasHidden);   
        outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);

       Label = Output(i,:);
       tarVector = answers(:,Label+1);
       myError = tarVector - outOfOutput;

       S2 = -2.*diag(ones(size(outOfOutput))-outOfOutput.*outOfOutput)*myError;   
       S1 = diag(ones(size(outOfHidden))-outOfHidden.*outOfHidden)*OutputLayerWeights'*S2;

       OutputLayerWeights = OutputLayerWeights - LearningRate * S2 * outOfHidden';  
       HiddenLayerWeights = HiddenLayerWeights - LearningRate * S1 * Input(:,i)';

       biasOutput = biasOutput - LearningRate.*S2;
       biasHidden = biasHidden - LearningRate.*S1;

       xVec(IterationCount) = IterationCount;
       % ErrorVec1(IterationCount) = sum(myError.^2)/length(myError);
       ErrorVecMaster(thisEpoch,IterationCount) = sum(myError.^2)/length(myError);
       tester = sum(myError.^2);    
    end
    HiddenLayerWeights = (b-a).*rand(NumHidLayerNeurons,784) + a; % Weight matrix from Input to Hidden
    OutputLayerWeights = (b-a).*rand(NumOutLayer,NumHidLayerNeurons) + a; % Weight matrix from Hidden to Output
    biasHidden = (b-a).*rand(NumHidLayerNeurons,1) + a;         % Random bias.
    biasOutput = (b-a).*rand(NumOutLayer,1) + a;
    IterationCount = 0;
    thisEpoch = thisEpoch + 1
    LearningRate = LearningRate * 2;
end
    

% figure(1)
% plot(xVec,ErrorVec1)
% title('Backpropagation Network Training')
% xlabel('Backpropagation Iterations')
% ylabel('Squared Error')

for a = 1:Epochs
    figure(a)
    plot(xVec,ErrorVecMaster(a,:))
    title('Backpropagation Network Training for Variable Learning')
    xlabel('Backpropagation Iterations')
    ylabel('Squared Error')
end
    

function res = mySoftMax(n)
    res = exp(n)/sum(exp(n));
end