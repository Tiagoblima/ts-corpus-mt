! onmt_build_vocab -config brnn.gn-pt.yaml -n_sample 10000
! onmt_train -config brnn.arc-ntlh.yaml
! onmt_translate -model run/gn-pt/model_step_10000.pt -src ../dataset/arc-ntlh/test.arc -output prediction/pred_arc-.txt -gpu 0 -verbose

! onmt_build_vocab -config brnn.gaj-pt.yaml -n_sample 10000
! onmt_train -config brnn.gaj-pt.yaml
! onmt_translate -model run/gaj-pt/model_step_10000.pt -src ../dataset/guajajara-portuguese/test.guajajara -output prediction/pred_gaj-pt.txt -gpu 0 -verbose

! onmt_build_vocab -config brnn.ka-pt.yaml -n_sample 10000
! onmt_train -config brnn.ka-pt.yaml
! onmt_translate -model run/ka-pt/model_step_10000.pt -src ../dataset/karaja-portuguese/test.karaja -output prediction/pred_ka-pt.txt -gpu 0 -verbose
