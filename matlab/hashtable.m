name = "veach-ajar/scene";
dirname = "/home/xd/Research/pathrenderer/scenes/";

% str_name_h = sprintf("%s%s_clusters.bin", dirname, name);
str_cluster = sprintf("%s%s_new_sub_clusters_2.bin", dirname, name);
str_cluster_2 = sprintf("%s%s_new_sub_clusters_idx.bin", dirname, name);
cID = fopen(str_cluster);
% hashtableID = fopen(str_name_h);
cID2 = fopen(str_cluster_2);

% hasht = fread(hashtableID, 'int32');
clusters = fread(cID, 'int32');
clusters2 = fread(cID2, 'int32');

outputname = sprintf("%s%s", dirname, name);

% drawhashbar(outputname, hasht, 'cluster1');
drawhashbar(outputname, clusters, 'cluster2');
drawhashbar(outputname, clusters2, 'clusters');
% drawsubclusters(outputname, clusters);
% drawhistogram(outputname, clusters2, clusters, hasht);

function drawhistogram(filename, table1, table2, table3)
    fig = figure('visible', 'off');
    h1 = histogram(table1, 'FaceColor', '#0072BD');
    hold on;
    h2 = histogram(table2, 'FaceColor', '#D95319');
    hold on;
    h3 = histogram(table3, 'FaceColor', '#EDB120');
    legend([h1,h2,h3],{'subdivide 2','subdivide 1','no subdivide'});
    % legend({'f(x) = (x * 5039 + 39916801)% size\_of\_table'},'Location','north')
    xlabel('#cluster points ');
    ylabel('#cluster');
    title('Points Distribution in Clusters');
    str_n = sprintf('%s_%s_histogram.jpg', filename, 'all');
    saveas(fig, str_n);
    close(fig);
end

function drawhashbar(filename, table, name) 
    xdim = 1:100;
    % disp(xdim);
    % disp(table(1:30));
    fig = figure('visible', 'off');
    plot(xdim,table(1:100));
    % legend({'f(x) = (x * 5039 + 39916801)% size\_of\_table'},'Location','north')
    legend({'f(x) = burtle(x) % size\_of\_table'},'Location','north')
    xlabel('#cluster points ');
    ylabel('#cluster');
    title('Hash Table Points Distribution');
    str_n = sprintf('%s_%s_histogram.jpg', filename, name);
    saveas(fig, str_n);
    close(fig);

    % fig  = figure('visible', 'off');
    % bar(xdim, table);
    % % legend({'f(x) = (x * 5039 + 39916801)% size\_of\_table'},'Location','north')
    % legend({'f(x) = burtle(x) % size\_of\_table'},'Location','north')
    % xlabel('#cluster idx ');
    % ylabel('#cluster points');
    % title('Hash Table Points Distribution');
    % str_n = sprintf('%s_%s_bar.jpg', filename, name);
    % saveas(fig, str_n);
    % close(fig);
end

function drawclusters(filename, table) 
    xdim = 1:length(table);

    fig = figure('visible', 'off');
    histogram(table);
    legend({'k=16'},'Location','northeast');
    xlabel('#points in cluster');
    ylabel('#cluter');
    title('cluster size Distribution');
    str_n = sprintf('%s_cluster_histogram.jpg', filename);
    saveas(fig, str_n);
    close(fig);

    fig  = figure('visible', 'off');
    bar(xdim, table);
    legend({'k=16'},'Location','northeast');
    xlabel('cluster idx ');
    ylabel('#point in cluster');
    title('cluster sizes');
    str_n = sprintf('%s_cluster_bar.jpg', filename);
    saveas(fig, str_n);
    close(fig);
end

function drawsubclusters(filename, table) 
    xdim = 1:length(table);

    fig = figure('visible', 'off');
    histogram(table);
    legend({'k=16'},'Location','northeast');
    xlabel('#sub clusters in cluster');
    ylabel('#cluter');
    title('sub cluster Distribution');
    str_n = sprintf('%s_cluster_histogram.jpg', filename);
    saveas(fig, str_n);
    close(fig);

    fig  = figure('visible', 'off');
    bar(xdim, table);
    legend({'k=16'},'Location','northeast');
    xlabel('cluster idx ');
    ylabel('#sub clusters in cluster');
    title('sub cluster');
    str_n = sprintf('%s_cluster_bar.jpg', filename);
    saveas(fig, str_n);
    close(fig);
end