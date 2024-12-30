clc
clear all
close all
%%
I1=imread("kodim07.jpg");
I1 = imresize(I1, [512 512]);
I1=rgb2gray(I1);
figure, imshow(I1)
title('original image');
figure,imhist(I1);
title('Histogram: original image');
imwrite(I1,'I1.jpg');
%I=imresize(x,28);
I1;
size(I1);
Row=size(I1,1);
Col = size(I1,2);
I=zeros(Row,Col);
I=I1(1:Row,1:Col);
I=int32(I);
%%
size(I);
size(I1)
Row=size(I,1)
Col = size(I,2)
%% Row 1 differences
D(1,1)=I(1,1);
  for j = 2: Col
   D(1,j) = I(1,j)-I(1,j-1);
  end
  I(1,2)-I(1,1);
D;
D=int32(D);
D=double(D);
%%
%%Retrieve row 1
D2(1,1)=D(1,1);
  for j = 2: Col
   D2(1,j)= D2(1,j-1)+D(1,j);
  end
D2;
D2=int32(D2);

%% Column Wise Differences
for i=1: Col
    for j = 2: Row
     D(j,i)= I(j,i)-I(j-1,i);
    end
end
 % Residual row image
figure,imshow(D);
title('Residual row image');
imwrite(D, 'RRI.jpg');
%% Column 1 differences
C1(1,1)=I(1,1);
C1=double(C1);
  for j = 2: Row
   C1(j,1) = I(j,1)-I(j-1,1);
  end
  I(1,2)-I(1,1);

%%
%%Retrieve column 1
C2(1,1)=D(1,1);
  for j = 2: Row
   C2(j,1)= C2(j-1,1)+C1(j,1);
  end
C2;
C2=int32(C2);
%% Row Wise Differences
for i=1: Row
    for j = 2: Col
     C1(i,j)= I(i,j)-I(i,j-1);
    end
end
C1% Residual column matrix
figure,imshow(C1);
title('Residual column image');
imwrite(C1, 'RCI.jpg');

%% Combine D and C1 by taking the Averager of D and C1 to form the combined residual matrix.
for i=1:Row
    for j=1:Col
        CD = fix((D(i,j)+C1(i,j))/2);
    end
end
CD=double(CD);
CD=(D+C1)/2;
%%

x=CD;
 imshow(x);
 
figure,imshow(x);
title('Difference Array');
size(x)
global M;
global N;
[M, N, d] = size(x);

H=[0.125 0.125 0.25 0 0.5 0 0 0;
0.125 0.125 0.25 0 -0.5 0 0 0;
0.125 0.125 -0.25 0 0 0.5 0 0;
0.125 0.125 -0.25 0 0 -0.5 0 0;
0.125 -0.125 0 0.25 0 0 0.5 0;
0.125 -0.125 0 0.25 0 0 -0.5 0;
0.125 -0.125 0 -0.25 0 0 0 0.5
0.125 -0.125 0 -0.25 0 0 0 -0.5];
 
 
A= zeros(8,8);
A1 = zeros(M,N);

count=1;
 
while count<=1
y=x(:,:,count);
x2=[];
a1=1;
while a1<=(M-7)
    a2=1;
   while a2<=(N-7) 
       A=y(a1:a1+7,a2:a2+7);
  
    A11=im2double(A);
    A =A11*H;
    A12=H'*A;
    if count == 1
    A1(a1:a1+7,a2:a2+7)=A12;
    end
 
   
a2=a2+8;
   end
 
a1=a1+8;
end

    count=count+1;

end
 size(x2);

s=0;
 y11=A1(:,:,1);
 A1;
 n1=nnz(y11);
 s=s+n1;
 mn = mean(A1);
 M_sum=0;
 for i=1:N
     M_sum = M_sum + mn(i);
 end
 av=M_sum/N;
 av;
     
  for i=1:M
       for j=1:N
          
           if A1(i,j)>= -av && A1(i,j)<= av
               A1(i,j)=0;
               
           end
       end
 end
  
figure,imshow(A1)
title('Wavelet transformed CD Array')
figure,imhist(A1);
title('Histogram: Wavelet transformed CD Array');
binranges = 0:1;
[bincounts] = histc(A1,binranges);

  cr1=nnz(y)/nnz(A1); 
  %% Band separate CD array
 %separating averages
   SA=[];

  N1=N/2;
  M1=M/2;
  N2=N1/4;
  M2=M1/4;

   for i=1:N2
       for j=1:M2
          
           i1=((i-1)*8);
           j1=((j-1)*8);
           
            for k=1:4
               for l=1:4
           SA(((i-1)*4)+k,((j-1)*4)+l)=A1(i1+k,j1+l);  
       end
            end
           
            for k=1:4
               for l=1:4
                 SA(((i-1)*4)+k,(((j-1)*4)+l)+M1)=A1(i1+k,(j1+l)+4); 
           
             end
           end
            
            for k=1:4
               for l=1:4
                    SA((((i-1)*4)+k)+N1,((j-1)*4)+l)=A1((i1+k)+4,j1+l); 
           
       end
            end
            
            for k=1:4
               for l=1:4
         
                 SA((((i-1)*4)+k)+N1,(((j-1)*4)+l)+M1)=A1((i1+k)+4,(j1+l)+4); 
                end
            end
            
           
           
       end
   end
   
   
   size(SA);
   
   figure,imshow(SA);
   title('Haar Transformed Band Separated CD array');
   imwrite(SA, 'SA.jpg');

   %%
   [X,Y,z1] = size(SA);
   disp(X)
   disp(Y)
   mid_col=Y/2;
   mid_row=X/2;
   % Define the four quadrants based on the midpoints
LL = SA(1:mid_row, 1:mid_col);
HL = SA(1:mid_row, mid_col+1:M);
LH = SA(mid_row+1:end, 1:mid_col);
HH = SA(mid_row+1:N, mid_col+1:M);
figure,imshow(LL)
title('LL Band of haar transformed CD array');
figure,imshow(HL)
title('HL Band of haar transformed CD array');
figure,imshow(LH)
title('LH Band of haar transformed CD array');
figure,imshow(HH)
imwrite(HH, 'HH.jpg');
title('HH Band of haar transformed CD array');
%% PCA Implementation
data = [reshape(LL, 1, 65536); reshape(HL, 1, 65536); reshape(LH, 1, 65536); reshape(HH, 1, 65536)];
[coeff, score, latent, tsquared] = pca(data, 'NumComponents', 1);

% 'coeff' contains the coefficients for each PC, 'score' contains the projections of the data onto the PCs,
% 'latent' contains the variances explained by each PC, and 'tsquared' provides information for hypothesis testing.
reduced_image = coeff(:, 1) * score(1, :);
reduced_image = reshape(reduced_image, 256, 256);
figure,imshow(reduced_image);
title('PCA reduced image');

%% Compute entropy of PCA reduced image the reduced image substitutes all the bands hence original size can be considered.

[r,c]=size(reduced_image);
prob=imhist(reduced_image)/(512*512);
prob=prob(prob>0);
Entropy_PCAreduced_image=-sum(prob.*log2(prob))

%%

%% LL Quantisation
% Assuming data is stored in a matrix named 'LL'. perform PCA
[coeff, score, latent] = pca(LL);
LL_Var=latent(1:255,1);
sortedLL = sort(LL_Var, 'descend');
middleIndex = floor(length(sortedLL) /2) + 1;  % Calculate middle index
mid_LL = sortedLL(middleIndex);  % Access middle element
Thresh_LL = mid_LL
LL_Quantized=LL;
LL_Quantized;
nnz(LL_Quantized);
%% HL Quantisation
% Assuming data is stored in a matrix named 'HL'. perform PCA
[coeff, score, latent] = pca(HL);
HL_Var=latent(1:255,1);
sortedHL = sort(HL_Var, 'descend');
middleIndex = floor(length(sortedHL) / 2) + 1;  % Calculate middle index
mid_HL = sortedHL(middleIndex);  % Access middle element
Thresh_HL = mid_HL
HL_Quantized=HL;
HL_Quantized;
nnz(HL_Quantized);

%% LH Quantisation
% Assuming data is stored in a matrix named 'LL'. perform PCA
[coeff, score, latent] = pca(LH);
LH_Var=latent(1:255,1);
sortedLH = sort(LH_Var, 'descend');
middleIndex = floor(length(sortedLH) / 2) + 1;  % Calculate middle index
mid_LH = sortedLH(middleIndex);  % Access middle element
Thresh_LH = mid_LH
LH_Quantized=LH;
LH_Quantized;
nnz(LH_Quantized);
%%
%% HH Quantisation
% Assuming your data is stored in a matrix named 'HH'. perform PCA
[coeff, score, latent] = pca(HH);
HH_Var=latent(1:255,1);
sortedHH = sort(HH_Var, 'descend');
middleIndex = floor(length(sortedHH) / 2) + 1;  % Calculate middle index
mid_HH = sortedHH(middleIndex);  % Access middle element
Thresh_HH = mid_HH
HH_Quantized=HH;
for i=1:256
     for j=1:256
          
        if HH_Quantized(i,j)<= Thresh_HH
             HH_Quantized(i,j)=0;
        end
     end 
end   
HH_Quantized;
imwrite(HH_Quantized, 'HH_Quantized.jpg');
nnz(HH_Quantized);
%%
%% Merge the bands to get quantized CD array
mid_row=256;
mid_col=256;
quantized_CD(1:mid_row, 1:mid_col)=LL_Quantized;
quantized_CD(1:mid_row, mid_col+1:M)= HL_Quantized;
quantized_CD(mid_row+1:N, 1:mid_col)= LH_Quantized;
quantized_CD(mid_row+1:N, mid_col+1:M)= HH_Quantized;
figure,imshow(quantized_CD)
title('quantized_CD image');
imwrite(quantized_CD, 'quantized_CD.jpg');
%HH_Quantized;
%% Perform inverse subband coding of quantized_CD image
 ISB=zeros(N,M);
   for i=1:N2
       for j=1:M2
          
           i1=((i-1)*8);
           j1=((j-1)*8);
           
            for k=1:4
               for l=1:4
          ISB(i1+k,j1+l)= quantized_CD(((i-1)*4)+k,((j-1)*4)+l);  
       end
            end
           
            for k=1:4
               for l=1:4
                ISB(i1+k,(j1+l)+4)= quantized_CD(((i-1)*4)+k,(((j-1)*4)+l)+M1); 
           
             end
           end
            
            for k=1:4
               for l=1:4
                  ISB((i1+k)+4,j1+l)= quantized_CD((((i-1)*4)+k)+N1,((j-1)*4)+l); 
           
       end
            end
            
            for k=1:4
               for l=1:4
         
                ISB((i1+k)+4,(j1+l)+4)= quantized_CD((((i-1)*4)+k)+N1,(((j-1)*4)+l)+M1); 
                end
            end
            
           
           
       end
   end
   
   
   %size(SA);
 figure,imshow(ISB);
  title('quantized_CD Inverse band transformed');
  imwrite(ISB, 'ISB.jpg');
  
   %%
%%
c=H';
c2=inv(H);
c1=inv(c);
count=1;
A =zeros(8,8);
A11 = zeros(M,N);

s=0;
while count<=1
 a1=1;   
y2=ISB(1:M,1:N,count);
x4=[];
 while a1<=(M-7)
    a2=1;
   while a2<=(N-7)
       A=y2(a1:a1+7,a2:a2+7);
       
  
    A13=c1*A*c2;
  

    if count == 1
    A11(a1:a1+7,a2:a2+7)=A13;
    
    end

 
a2=a2+8;
   end
   
  
   
a1=a1+8;
end

 
 
   count=count+1;
 
end
 
 
 

 imwrite(A11,'grayscalehaarCD.jpg');

figure, imshow('grayscalehaarCD.jpg');
title('Reconstructed Difference Array');
 

g=A11;
dr = int32(x) - int32(g);
dr;
mse = mean(abs(dr(:)).^2);
mse;
PSNR =   20 *( log10 (255 / sqrt(mse)));
PSNR;
 
CRE=nnz(x)/nnz(A11);
CRE
dr = int32(x) - int32(g);
dr;
mse1 = mean(abs(dr(:)).^2);
mse1;
PSNR1 =   20 *( log10 (255 / sqrt(mse1)));
PSNR1;


score_CD= multissim(x,g);
ssimval_CD = ssim(x,g);


%%
% Retrieve I from inverse wavelet transformed array A11
I3(1,1)=A11(1,1);
I3=int32(I3);
I3=double(I3);
A11=double(A11);
  for j = 2: Col
  I3(1,j)= A11(1,j)+I3(1,j-1);
  end
  for j = 2: Row
  I3(j,1)= A11(j,1)+I3(j-1,1);
  end
  I3;
  for i=2: Row
    for j = 2: Col
     I3(i,j)= A11(i,j) +(I3(i,j-1)+I3(i-1,j))/2;
    end
  end

I3=uint8(I3);
 I22=uint8(I3);
imwrite(I22,'I33.jpg');
figure,imshow(I22);
title('Reconstructed I22 after inverse wavelet transform');
figure,imshow('I33.jpg');
title('Reconstructed image from CD array after inverse wavelet transform');
I3;
isequal(I1,I3)
figure,imhist(I3);
title('Historam: Reconstructed wavelet transformed image');
%% Method 1
grayImage=rgb2gray(imread("kodim07.jpg"));
[r,c]=size(grayImage);
%figure,imhist(I)
prob=imhist(grayImage)/(r*c);
prob=prob(prob>0);
Entropy_I=-sum(prob.*log2(prob))
%%


%% Method 1
%grayImage=rgb2gray(imread('lena.jpg'));
[r,c]=size(I3);
%figure,imhist(I3)
prob=imhist(I3)/(r*c);
prob=prob(prob>0);
Entropy_I3=-sum(prob.*log2(prob))
%%
%% Method 1
%grayImage=rgb2gray(imread('lena.jpg'));
[r,c]=size(CD)
%figure,imhist(D)
prob=imhist(CD)/(r*c);
prob=prob(prob>0);
Entropy_CD=-sum(prob.*log2(prob))
%%
% Entropy of A11
%% Method 1
%grayImage=rgb2gray(imread('lena.jpg'));
[r,c]=size(A1)
%figure,imhist(D)
prob=imhist(A1)/(r*c);
prob=prob(prob>0);
Entropy_A1=-sum(prob.*log2(prob))
%%
imfinfo("kodim07.jpg")
peaksnr = psnr(I1,I3);
fprintf('\n The Peak-SNR value is %0.4f', peaksnr);
Img=imread('I33.jpg');
MSE = sum(sum((I1-Img).^2))/(Row*Col);
PSNR = 10*log10(256*256/MSE);
fprintf('\nMSE: %7.2f ', MSE);
fprintf('\nPSNR: %9.7f dB', PSNR);
%%
MSE = sum(sum((I1-I3).^2))/(Row*Col);
PSNR = 10*log10(256*256/MSE);
fprintf('\nMSE of I3 and I: %7.2f ', MSE);
fprintf('\nPSNR of I3 and I: %9.7f dB', PSNR);
%%

in = imfinfo("kodim07.jpg")
%Compression_Ratio_1 =  in.FileSize/length(matrgb)

% Now compute compression ratio for a single image
fileInfo = dir("I33.jpg")
bytesOnDisk = [fileInfo.bytes]
% Let's list them all and also compute compression ratio.
for k = 1 : length(fileInfo)
  thisName = fullfile(fileInfo(k).folder, fileInfo(k).name);
  imageInfo = imfinfo(thisName);
  imageSize = imageInfo.Width * imageInfo.Height * imageInfo.BitDepth/8;
  fprintf('The size of %s in memory is %d bytes, the disk size is %d\n',...
    fileInfo(k).name, imageSize, imageInfo.FileSize);
  % Now compute compression ratio...
end

%size of file in memory/size of file in disc for a single image
CRfinal= imageSize/imageInfo.FileSize


%%

% Now compute compression ratio for original to reconstructed image
fileInfo = dir("I33.jpg")
bytesOnDisk = [fileInfo.bytes]
% Let's list them all and also compute compression ratio.
for k = 1 : length(fileInfo)
  thisName = fullfile(fileInfo(k).folder, fileInfo(k).name);
  imageInfo2 = imfinfo(thisName);
  imageSize2 = imageInfo2.Width * imageInfo2.Height * imageInfo2.BitDepth/8;
  fprintf('The size of %s in memory is %d bytes, the disk size is %d\n',...
    fileInfo(k).name, imageSize2, imageInfo2.FileSize);
  % Now compute compression ratio...
end
compression_Ratio_2 = imageSize/imageInfo2.FileSize
BPP2= 8/compression_Ratio_2



Iref1=imread("kodim07.jpg");
Iref2=rgb2gray(Iref1);
I4=imread('I33.jpg');
size(Iref2);
size(I4);

%score = multissim(Iref2,I4);
%ssimval = ssim(Iref2,I4)

score = multissim(I3,I1)
ssimval = ssim(I3,I1)

%% Computing compression ratio: size in bytes of original/reconstructed image and bpp
fileInfo_I1 = dir('I1.jpg')
bytesOnDisk_I1 = [fileInfo_I1.bytes]
fileInfo_I33 = dir('I33.jpg')
bytesOnDisk_I33 = [fileInfo_I33.bytes]
CR_size_bytes = bytesOnDisk_I1/bytesOnDisk_I33;
BPP_size_bytes = 8/CR_size_bytes
