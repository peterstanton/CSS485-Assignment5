clear
format long

% Change the filenames if you've saved the files under different names
% On some platforms, the files might be saved as 
% train-images.idx3-ubyte / train-labels.idx1-ubyte
Input = loadMNISTImages('train-images.idx3-ubyte'); %inputs
Output = loadMNISTLabels('train-labels.idx1-ubyte'); %outputs

testIn = loadMNISTImages('t10k-images.idx3-ubyte');
testOut = loadMNISTLabels('t10k-labels.idx1-ubyte');


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
  
NumHidLayerNeurons = 100;
NumOutLayer = 10;

a = -0.01;
b = 0.01;
% r = (b-a).*rand(1000,1) + a;
 
HiddenLayerWeights = (b-a).*rand(NumHidLayerNeurons,784) + a; % Weight matrix from Input to Hidden
OutputLayerWeights = (b-a).*rand(NumOutLayer,NumHidLayerNeurons) + a; % Weight matrix from Hidden to Output
biasHidden = (b-a).*rand(NumHidLayerNeurons,1) + a;         % Random bias.
biasOutput = (b-a).*rand(NumOutLayer,1) + a;

tarIterations = 60000;

Epochs = 10;

ErrorVecMaster(tarIterations,Epochs) = 0;
ErrorTestVec(1:10000) = 0;


IterationCount = 0;  %Count passes.
% ErrorVec1(1:tarIterations) = 0;
 xVec(1:tarIterations) = 0;
 testXvec(1:10000) = 0;
i = 0;
thisEpoch = 1;
LearningVec(1:Epochs) = 0;
for e = 1:Epochs
    while(IterationCount < tarIterations)
        LearningVec(thisEpoch) = LearningRate;
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
       ErrorVecMaster(IterationCount,thisEpoch) = sum(myError.^2)/length(myError);
       
       
    end
    for count = 1:10000  %run through test set.
        count
        outOfHidden = tanh(HiddenLayerWeights * testIn(:,count) + biasHidden);   
        outOfOutput = logsig(OutputLayerWeights * outOfHidden + biasOutput);
        Label = testOut(count,:);
        tarVector = answers(:,Label+1);
        myError = tarVector - outOfOutput;
        testXvec(count) = count;
        ErrorTestVec(1,count) = sum(myError.^2)/length(myError);
    end
        figure(thisEpoch + 100)
        plot(testXvec,ErrorTestVec)
        title('Network performance at learning')
        xlabel('Test Set Entry')
        ylabel('Squared Error')
        annotation('textbox',[.2 .5 .3 .3],'String',LearningRate,'FitBoxToText','on');
    
    
    HiddenLayerWeights = (b-a).*rand(NumHidLayerNeurons,784) + a; % Weight matrix from Input to Hidden
    OutputLayerWeights = (b-a).*rand(NumOutLayer,NumHidLayerNeurons) + a; % Weight matrix from Hidden to Output
    biasHidden = (b-a).*rand(NumHidLayerNeurons,1) + a;         % Random bias.
    biasOutput = (b-a).*rand(NumOutLayer,1) + a;
    IterationCount = 0;
    thisEpoch = thisEpoch + 1
    LearningVec(thisEpoch) = LearningRate;
    LearningRate = LearningRate * 1.5;
end
    

% figure(1)
% plot(xVec,ErrorVec1)
% title('Backpropagation Network Training')
% xlabel('Backpropagation Iterations')
% ylabel('Squared Error')

for blah = 1:Epochs
    figure(blah)
    plot(xVec,ErrorVecMaster(:,blah))
    title('Backpropagation Network Training for Variable Learning')
    xlabel('Backpropagation Iterations')
    ylabel('Squared Error')
    annotation('textbox',[.2 .5 .3 .3],'String',LearningVec(blah),'FitBoxToText','on');

end

function res = mySoftMax(n)
    res = exp(n)/sum(exp(n));
end