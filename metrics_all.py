import xlrd
import xlsxwriter


class Metrics:

    def mrr(l_brand, l_in, l_ist, l_score):  # MRR
        influencer_num = 797
        brand_num = 74
        max_len = influencer_num * brand_num
        index = 0
        index_1 = 0
        top_rank = []
        for j in range(0, brand_num):
            # *****
            # print(j)
            l = [] #score list
            for i in range(0, influencer_num):
                a = l_score[index]
                l.append(a)
                if ((i + 1) % influencer_num == 0 or (index + 1) == max_len):
                    index += 1
                    break
                index += 1
            l.sort()

            ll = []  # list of positive example ranking score
            for i in range(0, influencer_num):
                ist = l_ist[index_1]
                # print(index_1)
                score = l_score[index_1]
                # print(i)
                rank = l.index(score) # compute ranking by score
                ist = int(ist)
                if (ist == 1):# is positive example
                    ll.append(influencer_num - rank)

                if ((i + 1) % influencer_num == 0 or (index_1 + 1) == max_len):
                    # ******
                    # print(ll)
                    y = min(ll)
                    top_rank.append(y) # top rank positive example of a brand
                    index_1 += 1
                    break
                index_1 += 1
        mrr_score = 0.0
        for t in top_rank:
            mrr_score += (1 / t)
        mrr_score = mrr_score / len(top_rank)
        print('MRR:', mrr_score)
        return mrr_score

    def map(l_brand, l_in, l_ist, l_score):
        influencer_num = 797
        brand_num = 74
        max_len = influencer_num * brand_num
        index = 0
        index_1 = 0
        map_list = [] # ap of all brands

        for j in range(0, brand_num):
            # *****
            # print(j)
            l = [] #score list
            for i in range(0, influencer_num):
                a = l_score[index]
                l.append(a)
                if ((i + 1) % influencer_num == 0 or (index + 1) == max_len):
                    index += 1
                    break
                index += 1
            l.sort()

            ll = []  # list of positive example ranking score
            for i in range(0, influencer_num):
                ist = l_ist[index_1]
                # print(index_1)
                score = l_score[index_1]
                rank = l.index(score) # compute ranking by score
                ist = int(ist)
                if (ist == 1):# is positive example
                    ll.append(influencer_num - rank)

                if ((i + 1) % influencer_num == 0 or (index_1 + 1) == max_len):
                    ll.sort()
                    map_score = 0.0
                    for k in range(len(ll)):
                        map_score += ((k + 1) / ll[k])
                    map_score = map_score / len(ll)
                    map_list.append(map_score)
                    index_1 += 1
                    break
                index_1 += 1
        average_map = 0.0
        for m in map_list:
            average_map += m
        average_map = average_map / len(map_list) # map
        print('mAP:', average_map)
        return average_map

    def metrics(l_brand, l_in, l_ist, l_score):  # MedR,r@10,r@50
        influencer_num = 797
        brand_num = 74
        max_len = influencer_num * brand_num
        index = 0
        index_1 = 0
        r_10 = [] # recall@10
        r_50 = [] # recall@50
        medr = [] # top rank positive example of all brands, to compute medr

        for j in range(0, brand_num):
            # *****
            # print(j)
            l = [] #score list
            for i in range(0, influencer_num):
                a = l_score[index]
                l.append(a)
                if ((i + 1) % influencer_num == 0 or (index + 1) == max_len):
                    index += 1
                    break
                index += 1
            l.sort()
            ll = []  # list of positive example ranking score

            for i in range(0, influencer_num):
                ist = l_ist[index_1]
                # print(index_1)
                score = l_score[index_1]
                rank = l.index(score)# compute ranking by score
                ist = int(ist)
                if (ist == 1):# is positive example
                    ll.append(influencer_num - rank)

                if ((i + 1) % influencer_num == 0 or (index_1 + 1) == max_len):
                    # ******
                    # print(ll)
                    y = min(ll) #top rank
                    medr.append(y)
                    index_2 = 0
                    index_3 = 0
                    for a in ll:
                        if (a < 11):# top 10
                            index_2 += 1
                        if (a < 51):# top 50
                            index_3 += 1
                    r1 = index_2 / len(ll)
                    r2 = index_3 / len(ll)

                    r_10.append(r1)
                    r_50.append(r2)
                    index_1 += 1
                    break
                index_1 += 1

        # MedR
        median = 0.0
        medr.sort()
        med = 0  # median index
        if (len(medr) % 2 == 0):
            med = len(medr) / 2
        else:
            med = int(len(medr) / 2) + 1
        for xy in range(0, len(medr)):
            if (xy == (med - 1)):
                median = medr[xy]
        print('MedR:', median)

        r10 = 0.0
        for xy1 in r_10:
            r10 += xy1
        # r@10
        print('r@10:', r10 / len(r_10))

        r50 = 0.0
        for xy2 in r_50:
            r50 += xy2
        # r@50
        print('r@50:', r50 / len(r_50))
        return median, r10 / len(r_10), r50 / len(r_50)

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