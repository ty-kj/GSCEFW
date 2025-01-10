% demo_AWNLRR_once
pp=sum(S.*S,2);
imshow(reshape(pp,32,32)*255);
pp_n=pp/sum(pp);

figure;
bar(pp_n)
l=gca;
l.YAxis.Exponent =-2;
xlabel('Feature Number')
ylabel('Weights')


figure;
imshow(reshape(pp_n*255,32,32));
figure;
imshow(reshape(pp_n*255,32,32),[]); %colormap jet; colorbar;
% imagesc(reshape(pp_n,32,32))


