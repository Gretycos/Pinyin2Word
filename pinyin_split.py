class PinyinSplit:
    def __init__(self):
        self.pinyin = {
            'a':['a','ai','an','ang','ao'],
            'b':['ba', 'bai', 'ban', 'bang', 'bao', 'bei', 'ben', 'beng', 'bi', 'bian', 'biao', 'bie', 'bin', 'bing', 'bo', 'bu'],
            'c':['ca', 'cai', 'can', 'cang', 'cao', 'ce', 'ceng', 'cha', 'chai', 'chan', 'chang',
                 'chao', 'che', 'chen', 'cheng', 'chi', 'chong', 'chou', 'chu', 'chuai', 'chuan',
                 'chuang', 'chui', 'chun', 'chuo', 'ci', 'cong', 'cou', 'cu', 'cuan', 'cui', 'cun', 'cuo'],
            'd':['da', 'dai', 'dan', 'dang', 'dao', 'de',
                'deng', 'di', 'dian', 'diao', 'die', 'ding', 'diu',
                'dong', 'dou', 'du', 'duan', 'dui', 'dun', 'duo','dia'],
            'e':['e', 'en', 'er','ei'],
            'f':['fa', 'fan', 'fang', 'fei', 'fen', 'feng', 'fu', 'fou','fo'],
            'g':['ga', 'gai', 'gan', 'gang', 'gao', 'ge', 'gei', 'gen', 'geng',
                'gong', 'gou', 'gu', 'gua', 'guai', 'guan', 'guang', 'gui', 'gun', 'guo'],
            'h':['ha', 'hai', 'han', 'hang', 'hao', 'he',
                'hei', 'hen', 'heng', 'hong', 'hou', 'hu',
                'hua', 'huai', 'huan', 'huang', 'hui', 'hun', 'huo'],
            'i':[],
            'j':['ji', 'jia', 'jian', 'jiang', 'jiao', 'jie', 'jin', 'jing', 'jiong',
                'jiu', 'ju', 'juan', 'jue', 'jun','jv'],
            'k':['ka', 'kai', 'kan', 'kang', 'kao', 'ke', 'ken', 'keng', 'kong', 'kou', 'ku', 'kua', 'kuai',
                'kuan', 'kuang', 'kui', 'kun', 'kuo'],
            'l':['la', 'lai', 'lan', 'lang', 'lao',
                'le', 'lei', 'leng', 'li', 'lia', 'lian', 'liang', 'liao', 'lie', 'lin',
                'ling', 'liu', 'long', 'lou', 'lu', 'luan', 'lue', 'lun', 'luo','lv','lve'],
            'm':['ma', 'mai', 'man', 'mang', 'mao', 'me', 'mei', 'men', 'meng', 'mi', 'mian',
                'miao', 'mie', 'min', 'ming', 'miu', 'mo', 'mou', 'mu'],
            'n':['na', 'nai', 'nan', 'nang', 'nao', 'ne', 'nei', 'nen', 'neng', 'ni', 'nian', 'niang',
                'niao', 'nie', 'nin', 'ning', 'niu', 'nong', 'nu', 'nuan', 'nue', 'nuo', 'nv'],
            'o':['o', 'ou'],
            'p':['pa', 'pai', 'pan', 'pang', 'pao', 'pei', 'pen',
                'peng', 'pi', 'pian', 'piao', 'pie', 'pin', 'ping', 'po', 'pou', 'pu'],
            'q':['qi', 'qia', 'qian', 'qiang', 'qiao', 'qie', 'qin', 'qing', 'qiong', 'qiu', 'qu',
                'quan', 'que', 'qun', 'qv'],
            'r':['ran', 'rang', 'rao', 're', 'ren', 'reng', 'ri',
                'rong', 'rou', 'ru', 'ruan', 'rui', 'run', 'ruo'],
            's':['sa', 'sai', 'san', 'sang', 'sao', 'se', 'sen', 'seng', 'sha', 'shai',
                'shan', 'shang', 'shao', 'she', 'shen', 'sheng', 'shi', 'shou', 'shu',
                 'shua', 'shuai', 'shuan', 'shuang', 'shui', 'shun', 'shuo', 'si',
                 'song', 'sou', 'su', 'suan', 'sui', 'sun', 'suo'],
            't':['ta', 'tai', 'tan', 'tang', 'tao', 'te', 'teng', 'ti', 'tian',
                'tiao', 'tie', 'ting', 'tong', 'tou', 'tu', 'tuan', 'tui', 'tun', 'tuo'],
            'u':[],
            'v':[],
            'w':['wa', 'wai', 'wan', 'wang', 'wei', 'wen', 'weng', 'wo', 'wu'],
            'x':['xi', 'xia', 'xian', 'xiang', 'xiao', 'xie', 'xin', 'xing', 'xiong', 'xiu', 'xu',
                'xuan', 'xue', 'xun', 'xv'],
            'y':['ya', 'yan', 'yang', 'yao', 'ye', 'yi', 'yin', 'ying',
                'yo', 'yong', 'you', 'yu', 'yuan', 'yue', 'yun'],
            'z':['za', 'zai', 'zan', 'zang', 'zao', 'ze', 'zei', 'zen', 'zeng', 'zha', 'zhai', 'zhan', 'zhang',
                'zhao', 'zhe', 'zhen', 'zheng', 'zhi', 'zhong', 'zhou', 'zhu', 'zhua',
                'zhuai', 'zhuan', 'zhuang', 'zhui', 'zhun', 'zhuo', 'zi', 'zong', 'zou', 'zu',
                'zuan','zui', 'zun', 'zuo'],
            '。':['。'],
            '；':['；'],
            '，': ['，'],
            '：': ['：'],
            '“': ['“'],
            '”': ['”'],
            '（': ['（'],
            '）': ['）'],
            '？': ['？'],
            '《': ['《'],
            '》': ['》'],
            '！': ['！'],
            '…': ['…'],
            '……': ['……'],
            '、': ['、'],
            '「': ['「'],
            '」': ['」'],
            '【': ['【'],
            '】': ['】'],
        }

    def split(self,source): # 输入拼音 输出分离后的结果
        result = []
        start = 0
        length = len(source)
        lastWrong = False
        count = 0
        while start < length:
            count += 1
            if count == 2000:
                print(source, result)
                break
            lastWrong = False
            first = source[start]
            step = 1
            tmp = source[start]
            for i in range(6):
                if start+i+1 > length:
                    break
                piece = source[start:start+i+1]
                if i == 0 and len(self.pinyin[first]) == 0:  # 非声母开头
                    lastWrong = True
                    # print(source,result,first)
                    result[-1] = result[-1][:-1]
                    start -= 1
                    break
                if first in self.pinyin:
                    if piece in self.pinyin[first]:
                        tmp = piece
                        step = i + 1
            if not lastWrong:
                try:
                    if (tmp == 'o' and len(piece) >1 and not self.hasComplete(piece[1:])) or (len(tmp) == 1 and tmp not in self.pinyin[tmp]) or (tmp == 'en' and len(result)>0 and result[-1][-1] in ['r','n','g']): # 哦了，u开头，可能
                        lastWrong = True
                        result[-1] = result[-1][:-1]
                        if len(result[-1]) == 0:
                            del result[-1]
                        start -= 1
                except:
                    print(source, result)
            if lastWrong:
                continue
            result.append(tmp)
            start += step
        return result

    def hasComplete(self,pinyin):
        first = pinyin[0]
        piece = first
        if len(pinyin) == 1:
            if first in self.pinyin:
                return True
        else:
            for p in pinyin[1:]:
                if first in self.pinyin:
                    if piece in self.pinyin[first]:
                        return True
                piece = piece + p
        return False
