function plotSystem(truT,truR,pfT,pfR,index)
    m = length(truT)+length(truR); n = 2;
    assert(m == length(pfT)+length(pfR));
    
    figure(2); hold on;
    
        subplot(m,n,2*(1:m)-1); hold on;
            scatter3(truT(1), truT(2), truT(3), 'b*');
            scatter3(pfT(1), pfT(2), pfT(3), 'r*');
            legend('True', 'Particle Filter');
            title('3D camera path (mm)');
            axis tight; box on; grid on;
        hold off; k = 2;
        
        subplot(m,n,k);hold on;
            scatter(index, truR(1), 'b*');
            scatter(index, pfR(1), 'r*');
            title(['Quaternion-w MSE=' num2str(immse(truR(1),pfR(1)),6)]);
            axis tight; box on;
        hold off; k = k + 2;
        subplot(m,n,k);hold on;
            scatter(index, truR(2), 'b*');
            scatter(index, pfR(2), 'r*');
            title(['Quaternion-x MSE=' num2str(immse(truR(2),pfR(2)),6)]);
            axis tight; box on;
        hold off; k = k + 2;
        subplot(m,n,k);hold on;
            scatter(index, truR(3), 'b*');
            scatter(index, pfR(3), 'r*');
            title(['Quaternion-y MSE=' num2str(immse(truR(3),pfR(3)),6)]);
            axis tight; box on;
        hold off; k = k + 2;
        subplot(m,n,k);hold on;
            scatter(index, truR(4), 'b*');
            scatter(index, pfR(4), 'r*');
            title(['Quaternion-z MSE=' num2str(immse(truR(4),pfR(4)),6)]);
            axis tight; box on;
        drawnow; hold off; k = k + 2;
        
        subplot(m,n,k);hold on;
            scatter(index, truT(1), 'b*');
            scatter(index, pfT(1), 'r*');
            title(['Translation-x MSE=' num2str(immse(truT(1),pfT(1)),6)]);
            axis tight; box on;
        hold off; k = k + 2;
        subplot(m,n,k);hold on;
            scatter(index, truT(2), 'b*');
            scatter(index, pfT(2), 'r*');
            title(['Translation-y MSE=' num2str(immse(truT(2),pfT(2)),6)]);
            axis tight; box on;
        hold off; k = k + 2;
        subplot(m,n,k); hold on;
            scatter(index, truT(3), 'b*');
            scatter(index, pfT(3), 'r*');
            title(['Translation-z MSE=' num2str(immse(truT(3),pfT(3)),6)]);
            axis tight; box on;
        hold off;
        
    hold off;
    drawnow;
end
