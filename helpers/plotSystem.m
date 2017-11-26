function plotSystem(truT,truR,pfT,pfR,index)
    truR = rotm2eul(truR);
    pfR = rotm2eul(pfR);
    m = 6; n = 2;
    
    figure(2); hold on;
    
        subplot(m,n,2*(1:m)-1); hold on;
            scatter3(truT(1), truT(2), truT(3), 'b*');
            scatter3(pfT(1), pfT(2), pfT(3), 'r*');
            legend('True', 'Particle Filter');
            title('3D camera path (mm)');
            axis tight; box on; grid on;
        hold off;
        
        subplot(m,n,2);hold on;
            scatter(index, truR(1), 'b*');
            scatter(index, pfR(1), 'r*');
            title('Euler rotation \alpha (radians)');
            axis tight; box on;
        hold off;
        subplot(m,n,4);hold on;
            scatter(index, truR(2), 'b*');
            scatter(index, pfR(2), 'r*');
            title('Euler rotation \beta (radians)');
            axis tight; box on;
        hold off;
        subplot(m,n,6);hold on;
            scatter(index, truR(3), 'b*');
            scatter(index, pfR(3), 'r*');
            title('Euler rotation \gamma (radians)');
            axis tight; box on;
        drawnow; hold off;
        
        subplot(m,n,8);hold on;
            scatter(index, truT(1), 'b*');
            scatter(index, pfT(1), 'r*');
            title('x-axis');
            axis tight; box on;
        hold off;
        subplot(m,n,10);hold on;
            scatter(index, truT(2), 'b*');
            scatter(index, pfT(2), 'r*');
            title('y-axis');
            axis tight; box on;
        hold off;
        subplot(m,n,12); hold on;
            scatter(index, truT(3), 'b*');
            scatter(index, pfT(3), 'r*');
            title('z-axis');
            axis tight; box on;
        hold off;
        
    hold off;
    drawnow;
end
