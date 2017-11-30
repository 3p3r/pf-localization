function plotSystem(vw,cameraParams,System,frame,entry,pfT,pfR,index)
    %{
        1   2   3   4
        5   6   7   8
        9   10  11  12
        13  14  15  16
        17  18  19  20
        21  22  23  24
        25  26  27  28
    %}
    
    m = 7; n = 4;
    figure(1);
    
    h = subplot(m,n,[23 24 27 28]); cla(h);
    [~,I] = sort(System.wp,'descend');
    top = 100; semilogy(System.wp(sort(I(1:top))));
    xlabel('Particles (n)');
    title(['Top ' num2str(top) ' weights']);
    axis tight; box on; grid on;
    
    h = subplot(m,n,[1 2 5 6 9 10]); cla(h);
    showMatchedFeatures(frame,frame,System.imagePoints,worldToImage(cameraParams,quat2rotm(pfR'),pfT,System.worldPoints));
    legend('Ground truth','Estimates');
    title('Innovations');
    axis tight;
    
    hold on;
        
        subplot(m,n,[13 14 17 18 21 22 25 26]); hold on;
        scatter3(entry.ExtrinsicsTranslationTrue(1), entry.ExtrinsicsTranslationTrue(2), entry.ExtrinsicsTranslationTrue(3), 'k.');
        scatter3(entry.ExtrinsicsTranslationEst(1), entry.ExtrinsicsTranslationEst(2), entry.ExtrinsicsTranslationEst(3), 'r.');
        scatter3(pfT(1), pfT(2), pfT(3), 'b.');
        legend('True', 'RANSAC', 'Particle Filter');
        title('3D camera path (world-units mm)');
        xlabel('x'); ylabel('y'); zlabel('z');
        axis tight; box on; grid on; view(3);
        hold off;
        
        subplot(m,n,3); hold on;
        scatter(index, entry.ExtrinsicsRotationTrue(1), 'k.');
        scatter(index, entry.ExtrinsicsRotationEst(1), 'r.');
        scatter(index, pfR(1), 'b.');
        title('Rotation quaternion-w');
        axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,4); hold on;
        scatter(index, entry.ExtrinsicsRotationTrue(2), 'k.');
        scatter(index, entry.ExtrinsicsRotationEst(2), 'r.');
        scatter(index, pfR(2), 'b.');
        title('Rotation quaternion-x');
        axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,7); hold on;
        scatter(index, entry.ExtrinsicsRotationTrue(3), 'k.');
        scatter(index, entry.ExtrinsicsRotationEst(3), 'r.');
        scatter(index, pfR(3), 'b.');
        title('Rotation quaternion-y');
        axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,8); hold on;
        scatter(index, entry.ExtrinsicsRotationTrue(4), 'k.');
        scatter(index, entry.ExtrinsicsRotationEst(4), 'r.');
        scatter(index, pfR(4), 'b.');
        title('Rotation quaternion-z');
        axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,[11 12]); hold on;
        scatter(index, entry.ExtrinsicsTranslationTrue(1), 'k.');
        scatter(index, entry.ExtrinsicsTranslationEst(1), 'r.');
        scatter(index, pfT(1), 'b.');
        title('Translation x');
        axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,[15 16]); hold on;
        scatter(index, entry.ExtrinsicsTranslationTrue(2), 'k.');
        scatter(index, entry.ExtrinsicsTranslationEst(2), 'r.');
        scatter(index, pfT(2), 'b.');
        title('Translation y');
        axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,[19 20]); hold on;
        scatter(index, entry.ExtrinsicsTranslationTrue(3), 'k.');
        scatter(index, entry.ExtrinsicsTranslationEst(3), 'r.');
        scatter(index, pfT(3), 'b.');
        title('Translation z');
        axis tight; box on; grid on;
        hold off;
        
    hold off;
    drawnow;
    
    writeVideo(vw, frame2im(getframe(gcf)));
end
