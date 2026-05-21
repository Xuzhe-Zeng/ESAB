function [I_loc_lin] = LinearLocalDemosaic_BlackFly23S6C(I)
%% Author: Alexander J. Myers; 1/25/2023; amyers2@andrew.cmu.edu

%Function returns nearest neighbor local demosaiced image (I_loc_lin) as a double

%For BlackFly 23S6C camera with 'rggb' pattern and 1200x1920 pixel FOV

I = double(I); %input image can be a double or uint16, which will then be converted to a double
I_loc_lin = (zeros(size(I,1),size(I,2),3));
[X_int,Y_int]= meshgrid(1:1:size(I,2),1:1:size(I,1));

%Red, Channel 1
I_loc_lin(:,:,1) = interp2(1:2:size(I,2),1:2:size(I,1),I(1:2:size(I,1),1:2:size(I,2)),X_int,Y_int); %Dimensions of image are [Y,X] NOT [X,Y]

% Green, Channel 2
G_nn=zeros(size(I));

%Green at red pixels
    G_nn(3:4:size(I,1),3:4:size(I,2)-1) = (I(2:4:size(I,1),3:4:size(I,2)-1)+I(3:4:size(I,1)-1,2:4:size(I,2)-1)+I(4:4:size(I,1),3:4:size(I,2))+I(3:4:size(I,1),4:4:size(I,2)))/4; %Red Pixels
    G_nn(3:4:size(I,1),5:4:size(I,2)-3) = (I(2:4:size(I,1),5:4:size(I,2)-3)+I(3:4:size(I,1)-1,4:4:size(I,2)-3)+I(4:4:size(I,1),5:4:size(I,2)-2)+I(3:4:size(I,1),6:4:size(I,2)-2))/4; %Red Pixels
    
    G_nn(5:4:size(I,1)-3,3:4:size(I,2)-1) = (I(4:4:size(I,1)-1,3:4:size(I,2)-1)+I(5:4:size(I,1)-1,2:4:size(I,2)-1)+I(6:4:size(I,1),3:4:size(I,2))+I(5:4:size(I,1),4:4:size(I,2)))/4; %Red Pixels
    G_nn(5:4:size(I,1)-3,5:4:size(I,2)-3) = (I(4:4:size(I,1)-1,5:4:size(I,2)-3)+I(5:4:size(I,1)-1,4:4:size(I,2)-3)+I(6:4:size(I,1),5:4:size(I,2)-2)+I(5:4:size(I,1),6:4:size(I,2)-2))/4; %Red Pixels
     
%Green at blue pixels
    G_nn(2:4:size(I,1)-1,2:4:size(I,2)-2) = (I(1:4:size(I,1)-1,2:4:size(I,2)-2)+I(2:4:size(I,1),1:4:size(I,2)-2)+I(2:4:size(I,1)-1,3:4:size(I,2)-1)+I(3:4:size(I,1)-1,2:4:size(I,2)-1))/4; %Blue Pixels
    G_nn(2:4:size(I,1)-1,4:4:size(I,2)-4) = (I(1:4:size(I,1)-1,4:4:size(I,2)-2)+I(2:4:size(I,1),3:4:size(I,2)-2)+I(2:4:size(I,1)-1,5:4:size(I,2)-1)+I(3:4:size(I,1)-1,4:4:size(I,2)-1))/4; %Blue Pixels
    
    G_nn(4:4:size(I,1)-1,2:4:size(I,2)-2) = (I(3:4:size(I,1)-3,2:4:size(I,2)-2)+I(4:4:size(I,1)-1,1:4:size(I,2)-2)+I(4:4:size(I,1)-1,3:4:size(I,2)-1)+I(5:4:size(I,1)-1,2:4:size(I,2)-1))/4; %Blue Pixels
    G_nn(4:4:size(I,1)-1,4:4:size(I,2)-4) = (I(3:4:size(I,1)-3,4:4:size(I,2)-2)+I(4:4:size(I,1)-1,3:4:size(I,2)-2)+I(4:4:size(I,1)-1,5:4:size(I,2)-1)+I(5:4:size(I,1)-1,4:4:size(I,2)-1))/4; %Blue Pixels
    
    G_nn(1:2:size(I,1),2:2:size(I,2)) = I(1:2:size(I,1),2:2:size(I,2));
    G_nn(2:2:size(I,1),1:2:size(I,2)) = I(2:2:size(I,1),1:2:size(I,2));
    I_loc_lin(:,:,2) = G_nn;

% Blue, Channel 3
I_loc_lin(:,:,3) = interp2(2:2:size(I,2),2:2:size(I,1),I(2:2:size(I,1),2:2:size(I,2),:),X_int,Y_int);

% A few test points to check the algorithm
R_test_13_16 = (I(13,15)+I(13,17))/2;
assert(round(R_test_13_16)==round(I_loc_lin(13,16,1)),'An error occured with the red linear NN demosaicing.')

G_test_3_3 = (I(2,3)+I(3,2)+I(4,3)+I(3,4))/4;
G_test_6_4 = (I(5,4)+I(6,5)+I(7,4)+I(6,3))/4;
assert(round(G_test_3_3)==round(I_loc_lin(3,3,2))&&round(G_test_6_4)==round(I_loc_lin(6,4,2)),'An error occured with the green linear NN demosaicing.')

B_test_7_7 = (I(8,8)+I(6,6)+I(8,6)+I(6,8))/4;
assert(round(B_test_7_7)==round(I_loc_lin(7,7,3)),'An error occured with the blue linear NN demosaicing.')

end