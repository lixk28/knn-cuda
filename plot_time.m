A=load("time.txt");
B=log(A./1000);
cuda=B(1:8,:);
cublas=B(9:16,:);
omp=B(17:end,:);
cudaSpeedup=A(17:end,:)./A(1:8,:);
cublasSpeedup=A(17:end,:)./A(9:16,:);
dimension=[1,4,16,64,256];
number=[256,512,1024,2048,4096,8192,16384,32768];
for i=1:8
    plot(dimension,cuda(i,:),'ro-',dimension,cublas(i,:),'g*-',dimension,omp(i,:),'b+-');
    xlabel('Dimension d')
    ylabel('Log computation time')
    title(strcat('Log computation time as a function of d for n =  ',num2str(number(i))))
    legend('CUDA','CUBLAS','OpenMP','location','best')
    I = frame2im(getframe(gcf)); 
    J = imresize(I, [1312, 1750], 'bicubic');
    imwrite(J,strcat('dimension_',num2str(i),'.png'));
end
for j=1:5
    plot(number,cuda(:,j),'ro-',number,cublas(:,j),'g*-',number,omp(:,j),'b+-');
    xlabel('Number of points n')
    ylabel('Log computation time')
    title(strcat('Log computation time as a function of n for d =  ',num2str(dimension(j))))
    legend('CUDA','CUBLAS','OpenMP','location','best')
    I = frame2im(getframe(gcf)); 
    J = imresize(I, [1312, 1750], 'bicubic');
    imwrite(J,strcat('number_',num2str(j),'.png'));
    
    plot(number(1:7),cudaSpeedup(1:7,j),'ro-',number(1:7),cublasSpeedup(1:7,j),'g*-');
    xlabel('Number of points n')
    ylabel('Speed up')
    title(strcat('Speed up between GPU methods and OpenMP as a function of n for d =  ',num2str(dimension(j))))
    legend('CUDA vs OpenMP','CUBLAS vs OpenMP','location','north')
    I = frame2im(getframe(gcf)); 
    J = imresize(I, [1312, 1750], 'bicubic');
    imwrite(J,strcat('speedup_',num2str(j),'.png'));
end
B=load("time_shared.txt");
cuda=B(1:8,:);
cuda_s=B(17:24,:);
cudaSpeedup=cuda./cuda_s;
plot(dimension,cudaSpeedup(2,:),'ro-',dimension,cudaSpeedup(4,:),'g*-',dimension,cudaSpeedup(6,:),'b+-',dimension,cudaSpeedup(8,:),'ch-');
xlabel('Dimension d')
ylabel('Speed up')
title('Speed up of using shared memory as a function of d for different n')
legend('n = 512','n = 2048','n = 8192','n = 32768','location','best')
I = frame2im(getframe(gcf)); 
J = imresize(I, [1312, 1750], 'bicubic');
imwrite(J, 'dimension_shared_speedup.png');
  
plot(number,cudaSpeedup(:,1),'ro-',number,cudaSpeedup(:,2),'g*-',number,cudaSpeedup(:,3),'b+-',number,cudaSpeedup(:,4),'ch-',number,cudaSpeedup(:,5),'kp-');
xlabel('Number of points n')
ylabel('Speed up')
title('Speed up of using shared memory as a function of n for different d')
legend('d = 1','d = 4','d = 16','d = 64','d = 256','location','best')
I = frame2im(getframe(gcf)); 
J = imresize(I, [1312, 1750], 'bicubic');
imwrite(J,'number_shared_speedup.png');

