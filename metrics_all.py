import xlrd
import xlsxwriter

class Metrics:
    
    
    def mrr(l_brand, l_in, l_ist, l_score): # MRR
        all_positive = 797
        brand_num = 74
        max_len = all_positive * brand_num
        index = 0
        index_1 = 0
    
        top_rank = []
    
        for j in range(0, brand_num):
            # *****
            # print(j)
            l = []
            for i in range(0, all_positive):
                a = l_score[index]
                l.append(a)
    
                if ((i + 1) % all_positive == 0 or (index + 1) == max_len):
                    index += 1
                    break
                index += 1
            l.sort()
            ll = []  # positive example ranking score
    
            for i in range(0, all_positive):
    
                ist = l_ist[index_1]
                # print(index_1)
                score = l_score[index_1]
                # print(i)
                rank = l.index(score)
    
                ist = int(ist)
    
                if (ist == 1):
                    ll.append(all_positive - rank)
    
                if ((i + 1) % all_positive == 0 or (index_1 + 1) == max_len):
                    # ******
                    # print(ll)
    
                    y = min(ll)
    
                    top_rank.append(y)
                    index_1 += 1
                    break
                index_1 += 1
        mrr_score = 0.0
        for t in top_rank:
            mrr_score += (1/t)
        mrr_score = mrr_score/len(top_rank)
        print('MRR:', mrr_score)
        return mrr_score
    
    def map(l_brand, l_in, l_ist, l_score):
        all_positive = 797
        brand_num = 74
        max_len = all_positive * brand_num
        index = 0
        index_1 = 0
        map_list = []
    
        for j in range(0, brand_num):
            # *****
            # print(j)
            l = []
            for i in range(0, all_positive):
                a = l_score[index]
                l.append(a)
    
                if ((i + 1) % all_positive == 0 or (index + 1) == max_len):
                    index += 1
                    break
                index += 1
            l.sort()
            ll = []  # positive example ranking score
    
            for i in range(0, all_positive):
    
                ist = l_ist[index_1]
                # print(index_1)
                score = l_score[index_1]
                rank = l.index(score)
    
                ist = int(ist)
    
                if (ist == 1):
                    ll.append(all_positive - rank)
    
                if ((i + 1) % all_positive == 0 or (index_1 + 1) == max_len):
                    ll.sort()
    
                    map_score = 0.0
                    for k in range(len(ll)):
                        map_score += ((k+1)/ll[k])
                    map_score = map_score/len(ll)
                    map_list.append(map_score)
    
                    index_1 += 1
                    break
                index_1 += 1
        average_map = 0.0
        for m in map_list:
            average_map += m
        average_map = average_map/len(map_list)
        print('mAP:', average_map)
        return average_map
    
    def metrics(l_brand, l_in, l_ist, l_score):  # MedR,p@10,p@50
        all_positive = 797
        brand_num = 74
        max_len = all_positive * brand_num
        index = 0
        index_1 = 0
    
        p_10 = []
        p_50 = []
        medr = []
    
        for j in range(0, brand_num):
            # *****
            # print(j)
            l = []
            for i in range(0, all_positive):
                a = l_score[index]
                l.append(a)
    
                if ((i + 1) % all_positive == 0 or (index + 1) == max_len):
                    index += 1
                    break
                index += 1
            l.sort()
            ll = []  # positive example ranking score
    
            for i in range(0, all_positive):
    
                ist = l_ist[index_1]
                # print(index_1)
                score = l_score[index_1]
                rank = l.index(score)
    
                ist = int(ist)
    
                if (ist == 1):
                    ll.append(all_positive - rank)
    
                if ((i + 1) % all_positive == 0 or (index_1 + 1) == max_len):
                    # ******
                    # print(ll)
    
                    y = min(ll)
    
                    medr.append(y)
                    index_2 = 0
                    index_3 = 0
                    for a in ll:
                        if (a < 11):
                            index_2 += 1
                        if (a < 51):
                            index_3 += 1
                    p1 = index_2 / len(ll)
                    p2 = index_3 / len(ll)
    
                    p_10.append(p1)
                    p_50.append(p2)
                    index_1 += 1
                    break
                index_1 += 1
    
        # MedR
        median = 0.0
        medr.sort()
        med = 0 # median index
        if (len(medr) % 2 == 0):
            med = len(medr) / 2
        else:
            med = int(len(medr) / 2) + 1
        for xy in range(0, len(medr)):
            if (xy == (med - 1)):
                median = medr[xy]
        print('MedR:', median)
    
        p10 = 0.0
        for xy1 in p_10:
            p10 += xy1
        # p@10
        print('p@10:', p10 / len(p_10))
    
        p50 = 0.0
        for xy2 in p_50:
            p50 += xy2
        # p@50
        print('p@50:', p50 / len(p_50))
        return median, p10 / len(p_10), p50 / len(p_50)
    
    def auc(l_brand, l_in, l_ist, l_score):
        # AUC cAUC
        test_auc_file_path = './testset_auc.xlsx'
        ExcelFile1 = xlrd.open_workbook(test_auc_file_path)
        sheet1 = ExcelFile1.sheet_by_index(0)
        err = 0
        AUC = 0.0
        AUC_all = 0.0
        cAUC = 0.0
        cAUC_all = 0.0
    
        dict_s = {}
    
        for i in range(0, len(l_brand)):
            a_ = l_brand[i]
            b_ = l_in[i]
            score = l_score[i]
            a = a_ + b_
            dict_s[a] = score
        for i in range(0, sheet1.nrows):
            # if(i%10000==0):
            # print(i)
            AUC_all += 1
            a = sheet1.cell(i, 0).value.encode('utf-8').decode('utf-8-sig')
            b = sheet1.cell(i, 1).value.encode('utf-8').decode('utf-8-sig')
            c = sheet1.cell(i, 2).value.encode('utf-8').decode('utf-8-sig')
            e = sheet1.cell(i, 4).value.encode('utf-8').decode('utf-8-sig')
            f = sheet1.cell(i, 5).value.encode('utf-8').decode('utf-8-sig')
            score1 = 0.0
            score2 = 0.0
            if a + b in dict_s.keys():
                score1 = dict_s[a + b]
            if a + e in dict_s.keys():
                score2 = dict_s[a + e]
    
            if (score1 == 0.0 or score2 == 0.0):
                err += 1
            if (c == f):
                cAUC_all += 1
            if (score1 > score2):
                AUC += 1
                if (c == f):
                    # print(score1,score2)
                    cAUC += 1
    
        print('AUC is:', AUC / AUC_all, AUC)
        print('cAUC is:', cAUC / cAUC_all, cAUC)
        print(err)
        return AUC / AUC_all, cAUC / cAUC_all
