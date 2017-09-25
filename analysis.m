[value,index]=max(res');
correct = find(index==lab);
% for i = 1:size(correct,2)
%     plot(res(correct(i),:))
%     pause
% end
wrong = find(index~=lab);
correct_res = res(correct,:);
wrong_res = res(wrong,:);

correct_res_sorted = sort(correct_res,2);
wrong_res_sorted=sort(wrong_res,2);

correct_res_717_716_ratio = correct_res_sorted(:,717)./correct_res_sorted(:,716);
wrong_res_717_716_ratio = wrong_res_sorted(:,717)./wrong_res_sorted(:,716);
plot(correct_res_717_716_ratio(find(correct_res_717_716_ratio<10),:))
hold on
plot(wrong_res_717_716_ratio)

mean(correct_res_717_716_ratio)
mean(wrong_res_717_716_ratio)


mean(correct_res_717_716_ratio(find(correct_res_717_716_ratio<10)))
mean(wrong_res_717_716_ratio(find(wrong_res_717_716_ratio<10)))