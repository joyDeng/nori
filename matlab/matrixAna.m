jdxfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrixJdx.bin');
idxfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrixIdx.bin');
pidxfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_pixel_idx.bin')

ARfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_r.bin');
AGfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_g.bin');
ABfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_b.bin');

rvaluefileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_b_value_r.bin')
gvaluefileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_b_value_g.bin')
bvaluefileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_b_value_b.bin')

rxinitfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_x_0_r_value.bin')
gxinitfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_x_0_g_value.bin')
bxinitfileID = fopen('/home/xd/Research/pathrenderer/scenes/living-room-2/scene_matrix_x_0_b_value.bin')

IDX = fread(jdxfileID, 'int32');
JDX = fread(idxfileID, 'int32');
PDX = fread(pidxfileID, 'int32');

AR = fread(ARfileID, 'float');
AG = fread(AGfileID, 'float');
AB = fread(ABfileID, 'float');

bR = fread(rvaluefileID, 'float');
bG = fread(gvaluefileID, 'float');
bB = fread(bvaluefileID, 'float');

X_0_r = fread(rxinitfileID, 'float');
X_0_g = fread(gxinitfileID, 'float');
X_0_b = fread(bxinitfileID, 'float');


pathlength = length(PDX);
M = length(X_0_r)



IDX = IDX + 1;
JDX = JDX + 2;

mask_1 = PDX == -2;
PDX = PDX + 1;
PDX(mask_1) = 1;

JDX_mask = JDX >= M
JDX(JDX_mask) = 1

R = PAmat(IDX, JDX, AR, M);
% G = PAmat(IDX, JDX, AG, M);
% B = PAmat(IDX, JDX, AB, M);

eigenv_R_D = eigs(R);
% eigenv_G_D = eigs(G);
% eigenv_B_D = eigs(B);
disp(eigenv_R_D)

% Ama = speye(M) - R
% determinantOfA = det(Ama)
% disp(determinantOfA)
% 
% eigenv_R = diag(eigenv_R_D);
% eigenv_G = diag(eigenv_G_D);
% eigenv_B = diag(eigenv_B_D);
% 
% mask_r = eigenv_R == 1.0;
% mask_g = eigenv_G == 1.0;
% mask_b = eigenv_B == 1.0;
% 
% mask_all = mask_r + mask_g + mask_b;
% mask_rgb = mask_all == 3.0;
% 
% Vector_R_relateToOne = vector_R_V(:, mask_rgb);
% Vector_G_relateToOne = vector_G_V(:, mask_rgb);
% Vector_B_relateToOne = vector_B_V(:, mask_rgb);
% 
% c = 2
% 
% vR = Vector_R_relateToOne(:,c);
% vG = Vector_G_relateToOne(:,c);
% vB = Vector_B_relateToOne(:,c);
% 
% P_V_R = pixelValue(vR, mask_1, PDX);
% P_V_G = pixelValue(vG, mask_1, PDX);
% P_V_B = pixelValue(vB, mask_1, PDX);
% 
% 
% t = 1:length(eigenv_B)
% t2 = 1:length(vR)

% fig = figure('visible', 'off');
% plot(t, eigenv_R, t, eigenv_G, t, eigenv_B);
% legend({'eigenv_R', 'eigenv_G', 'eigenv_B'},'Location','southwest')
 
% saveas(fig, "/home/xd/Research/pathrenderer/scenes/living-room-2/scene_eigen_values.jpg");
% close(fig)

% X_l_r = lsqnonneg(R, bR);
% X_l_g = lsqnonneg(B, bR);
% X_l_b = lsqnonneg(G, bR);

% p_X_0_r = pixelValue(X_0_r, mask_1, PDX);
% p_X_0_g = pixelValue(X_0_g, mask_1, PDX);
% p_X_0_b = pixelValue(X_0_b, mask_1, PDX);

% p_X_l_rt = pixelValue(X_l_r, mask_1, PDX);
% p_X_l_gt = pixelValue(X_l_g, mask_1, PDX);
% p_X_l_bt = pixelValue(X_l_b, mask_1, PDX);

% p_X_l_r = truncNegative(p_X_l_rt)
% p_X_l_g = truncNegative(p_X_l_gt)
% p_X_l_b = truncNegative(p_X_l_bt)

% widtht = 64
% height = 64

% filename1 = '/home/xd/Research/pathrenderer/scenes/cbox/cbox_diffuse-X_0.hdr'
% filename2 = '/home/xd/Research/pathrenderer/scenes/cbox/cbox_diffuse-X_l.hdr'
% filename3 = '/home/xd/Research/pathrenderer/scenes/cbox/cbox_diffuse-Vector.hdr'
% mywriteHDR(p_X_0_r, p_X_0_g, p_X_0_b, widtht, height, filename1)
% mywriteHDR(p_X_l_r, p_X_l_g, p_X_l_b, widtht, height, filename2)
% mywriteHDR(P_V_R, P_V_G, P_V_B, widtht, height, filename3)
% storeAllEigen(Vector_R_relateToOne, Vector_G_relateToOne, Vector_B_relateToOne, widtht, height, mask_1, '/home/xd/Research/pathrenderer/scenes/cbox/cbox_', PDX);

function A = Amat(id,jd, values, dimension)
    A = speye(dimension) - sparse(id, jd, values, dimension, dimension)
end

function PA = PAmat(id,jd, values, dimension)
    PA = sparse(id, jd, values, dimension, dimension)
end

function f = mywriteHDR(r, g, b, w, h, filename)
    R = reshape(r, h, w, 1);
    G = reshape(g, h, w, 1);
    B = reshape(b, h, w, 1);
    img = cat(3, R, G, B)
   hdrwrite(img, filename)
end

function pixl = pixelValue(values, mask, ppid)
    pixl = values(ppid);
    pixl(mask) = 0.0;
end

function a = truncNegative(b)
    a = b;
    mask = b < 0;
    a(mask) = 0;
end

function b = storeAllEigen(eR, eG, eB, w, h, msk, folder, pdx)
    for c = 1:length(eR)
        filename = sprintf('%s_%03d.hdr', folder, c);
        plotname = sprintf('%s_plot_%03d.png', folder, c);

        rff = eR(:,c);
        gff = eG(:,c);
        bff = eB(:,c);

        fig = figure('visible', 'off');
        
        t2 = 1:length(bff);
        plot(t2, rff, t2, gff, t2, bff);
        legend({'Red', 'Green', 'Blue'},'Location','northeast');
        saveas(fig, plotname);

        close(fig)

        r = pixelValue(rff, msk, pdx);
        g = pixelValue(gff, msk, pdx);
        b = pixelValue(bff, msk, pdx);

        mywriteHDR(r, g, b, w, h, filename)
    end
end