name = "veach-ajar/scene";
dirname = "/home/xd/Research/pathrenderer/scenes/";

str_name_jdx = sprintf("%s%s_matrixJdx.bin", dirname, name);
str_name_idx = sprintf("%s%s_matrixIdx.bin", dirname, name);
str_name_AR = sprintf("%s%s_matrix_r.bin", dirname, name);
str_name_AG = sprintf("%s%s_matrix_g.bin", dirname, name);
str_name_AB = sprintf("%s%s_matrix_b.bin", dirname, name);
str_name_b = sprintf("%s%s_matrix_b_value.bin", dirname, name);
str_name_x = sprintf("%s%s_matrix_x_0_value.bin", dirname, name);

str_name_matrix_info = sprintf("%s%s_matrix_info.bin", dirname, name);
str_name_img_info = sprintf("%s%s_img_info.bin", dirname, name);
str_name_cluster = sprintf("%s%s_clusters.bin", dirname, name);
str_name_pixel_idx = sprintf("%s%s_matrix_pixel_idx.bin", dirname, name);

jdxfileID = fopen(str_name_jdx);
idxfileID = fopen(str_name_idx);
ARfileID = fopen(str_name_AR);
AGfileID = fopen(str_name_AG);
ABfileID = fopen(str_name_AB);
xinitfileID = fopen(str_name_x);
bfileID = fopen(str_name_b);

infofileID = fopen(str_name_matrix_info);
clusterfileID = fopen(str_name_cluster);
pidxfileID = fopen(str_name_pixel_idx);
imginfoID = fopen(str_name_img_info);

eigenvfilename = sprintf("%s%s__eigenv_complex.jpg", dirname, name);

IDX = fread(idxfileID, 'int32');
JDX = fread(jdxfileID, 'int32');
PDX = fread(pidxfileID, 'int32');
clusters = fread(clusterfileID, 'int32');

Ar = fread(ARfileID, 'float');
Ag = fread(AGfileID, 'float');
Ab = fread(ABfileID, 'float');

X = fread(xinitfileID, 'float');
b = fread(bfileID, 'float');

X = reshape(X, 3, []);
b = reshape(b, 3, []);


dimension = fread(infofileID, 'int32');
wandh = fread(imginfoID, 'int32');

inputtname = sprintf("%s%s", dirname, name);
outputname = sprintf("%s%s_scene_output_d%d", dirname, name, dimension);

IDX = IDX + 1;
JDX = JDX + 1;

R = PAmat(IDX, JDX, Ar, dimension);
G = PAmat(IDX, JDX, Ag, dimension);
B = PAmat(IDX, JDX, Ab, dimension);

disp(size(b));
disp(dimension);
disp(size(IDX));

% spy(R);
% m = symamd(R);
% spy(R(m, m));
% title('Sparse Matrix Pattern');

% x = pow2(IDX, -32);
% y = pow2(JDX, -32);

% Rn = R + R';
% d = abs(sum(Rn))+1;
% Rn = Rn + diag(sparse(d));

% gplot(Rn, [x,y]);
% title('cross-section of R');

% IterAndSaveRGB(outputname, R,G,B, b, X, PDX, 180, wandh(1), wandh(2));

% % str_n = sprintf('%s_eigenvalues.jpg', outputname);
% [eigen_vectors, eigen_values] = saveComplexEigenValues(R, 1, str_n);
% saveData2File(eigen_vectors, eigen_values, PDX, outputname);

% saveEigenValues(outputname, eigen_vectors, PDX, wandh(1), wandh(2));
% writeEigenvalues2File(outputname, eigen_vectors)
% saveMaxIndex(outputname, eigen_vectors)
    
IterAndSaveRGB(inputtname, R, G, B, b, X, PDX, 3,wandh(1), wandh(2))
% IterClamp(inputtname, R, b, X, PDX, 3, wandh(1), wandh(2));

function IterClamp(filename, A, B, x, pid, N, w, h)
    xr = x(1, :).';
    br = B(1, :).';
    disp(length(br));
    r = [];
    rc = [];
    for c = 1:N
        str_f = sprintf('%s_%02d_X.hdr',filename, c-1);
        xr = Jacobian(A, xr, br);
        r = cat(1, r, norm(xr - br));
        saveImgRGB(str_f, xr-br, xr-br, xr-br, pid, w, h);

        str_f = sprintf("%s%d_X.bin", filename, c-1);
        disp(str_f);
        fileID = fopen(str_f);
        x_n = fread(fileID, 'float');
        str_n = sprintf("%s_%02d_X_clamp.hdr",filename, c-1);
        rc = cat(1, rc, norm(x_n));
        saveImgRGB(str_n, x_n, x_n, x_n, pid, w, h);
    end
    xdim = 1:N;

    fig = figure('visible', 'off');
    p = plot(xdim, r, xdim, rc);
    p(1).LineWidth = 1.5;
    p(2).LineWidth = 1.5;
    xlim([0, N])
    legend({'no clamp','clamp by ratio'},'Location','north')
    xlabel('iteration');
    ylabel('norm');
    title('Norm of MX through iterations');
    str_s = sprintf('%s_norm_wrt_iter.jpg', filename);
    saveas(fig, str_s);
    close(fig);
end

function output = saveMaxIndex(filename,ev)
    sizes = size(ev);
    str_f = sprintf("%s_max_idx.bin", filename);
    eigenvectorsID = fopen(str_f, 'w');
    id = [];
    for c = 1:sizes(2)
        [d, I] = max(ev(:,c));
        id = cat(1, id, I);
    end
    disp(id);
    fwrite(eigenvectorsID, id, 'int');
    fclose(eigenvectorsID);
end

function c = saveData2File(ev, eva, pid, filename)
    sizes = size(ev);
    mask = (pid == -2);
    pixel_idx = pid + 1;
    pixel_idx(mask) = 1;
    str_f = sprintf("%s_%d.txt", filename, sizes(2));
    eigenvectorsID = fopen(str_f, 'w');

    for c = 1:sizes(2)
        t = ev(:,c);
        t_1 = t(pixel_idx);
        t_1(mask) = 0.0;
        norm1 = norm(t_1);
        norm_full = norm(t);
        [m, I] = max(abs(t));
        str_n = sprintf("eigenvalue %d = %.8f first bounce norm = %.8f, norm = %.8f, maxvalue = %.8f, maxid = %d\n", c, eva(c), norm1, norm_full, m, I);
        fprintf(eigenvectorsID, "%s", str_n);
        disp(str_n);
    end
    fclose(eigenvectorsID);
end

function c = normEigenvalues(ev, pid)
    sizes = size(ev);
    mask = (pid == -2);
    pixel_idx = pid + 1;
    pixel_idx(mask) = 1;

    for c = 1:sizes(2)
        t = ev(:,c);
        t_1 = t(pixel_idx);
        t_1(mask) = 0.0;
        norm1 = norm(t_1);
        norm_full = norm(t);
        str_n = sprintf("norm_1: %.8f, norm_full: %.8f", norm1, norm_full);
        disp(str_n);
    end
end

function w = writeEigenvalues2File(outputname, ev)
    sizes = size(ev);
    for c = 1:sizes(2)
        str_n = sprintf('%s_ev_%02d.bin', outputname, c);
        fileID = fopen(str_n, 'w');
        fwrite(fileID, ev(:,c), 'float');
        fclose(fileID);
    end
end

% saveImg(outputname, output, PDX, wandh(1), wandh(2));
function pv = getValues(data, idx, mask)
    pv = data(idx);
    pv(mask) = 0.0;
end

function output = saveImgRGB(filename, xr, xg, xb, pdx, width, height)
    mask = (pdx == -2);
    pixel_idx = pdx + 1;
    pixel_idx(mask) = 1;

    r = getValues(xr, pixel_idx, mask);
    g = getValues(xg, pixel_idx, mask);
    b = getValues(xb, pixel_idx, mask);

    mywriteHDR(r, g, b, width, height, filename);
    output = true;
end



function IterAndSaveRGB(filename, R, G, B, b, x, pdx, N, width, height)
    xr = x(1, :).';
    xg = x(2, :).';
    xb = x(3, :).';
    br = b(1, :).';
    bg = b(2, :).';
    bb = b(3, :).';
    disp(length(br));
    r = [];
    g = [];
    b = [];
    for c = 1:N
        str_f = sprintf('%s_%02d.hdr',filename, c);
        xr = Jacobian(R, xr, br);
        xg = Jacobian(G, xg, bg);
        xb = Jacobian(B, xb, bb);
        r = cat(1, r, norm(xr - br));
        g = cat(1, g, norm(xg - br));
        b = cat(1, b, norm(xb - br));
        saveImgRGB(str_f, xr, xg, xb, pdx, width, height);
    end
    xdim = 1:N;

    fig = figure('visible', 'off');
    plot(xdim, r);
    xlim([0, N])
    % ylim([0  2.0]);
    xlabel('iteration');
    ylabel('norm');
    title('Norm of MX through iterations');
    str_s = sprintf('%s_norm_wrt_iter.jpg', filename);
    saveas(fig, str_s);
    close(fig);
    
end

function output = saveImg(filename, data, pdx, width, height)
    mask = (pdx == -2);
    pixel_idx = pdx + 1;
    pixel_idx(mask) = 1;

    pixels = data(pixel_idx);
    pixels(mask) = 0;

    mywriteHDR(pixels, pixels, pixels, width, height, filename);
    output = true;
end

function a = saveEigenValues(filename, vectors, pid, w, h)
    sizes = size(vectors);
    disp(sizes);
    col = sizes(2);
    disp(col);
    for id = 1:col
        fs = sprintf("%s_eigen_vectors_%02d.hdr", filename, id);
        disp(id);
        data = vectors(:,id);
        saveImg(fs, abs(data), pid, w, h);
    end
    a = true;
end

function output = IterAndSave(filename, A, b, x, pdx, N, width, height)
    for c = 1:N
        str_f = sprintf('%s_%02d.hdr',filename, c);
        x = Jacobian(A, x, b);
        saveImg(str_f, x, pdx, width, height);
    end
end


function  [t, v] = saveComplexEigenValues(matrixR, dim, filename)
    [vector_R_V, eigenv_R_D] = eigs(matrixR, dim);
    eigenv = diag(eigenv_R_D);
    real_eigenvs = real(eigenv);
    image_eigenvs = imag(eigenv);
    fig = figure('visible', 'off');
    plot(real_eigenvs, image_eigenvs, 'ro');
    viscircles([0.0 0.0], [1], "Color", 'b');
    viscircles([0.0 0.0], [2], "Color", 'b');
    viscircles([0.0 0.0], [3], "Color", 'b');
    xlim([-3  3])
    ylim([-3  3]);
    xlabel('real');
    ylabel('imag');
    
     
    saveas(fig, filename);
    close(fig);
    t = vector_R_V;
    v = eigenv;
end

function output = Iter(N, A, x, b)
    for c = 1:N
        x = Jacobian(A, x, b);
    end
    output = x;
end

function x1 = Jacobian(A, x, b)
    x1 = A * x + b;
end

function A = Amat(id,jd, values, dimension)
    A = speye(dimension) - sparse(id, jd, values, dimension, dimension);
end

function PA = PAmat(id,jd, values, dimension)
    PA = sparse(id, jd, values, dimension, dimension);
end

function f = mywriteHDR(r, g, b, w, h, filename)
    R = reshape(r, h, w, 1);
    G = reshape(g, h, w, 1);
    B = reshape(b, h, w, 1);
    img = cat(3, R, G, B);
   hdrwrite(img, filename);
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