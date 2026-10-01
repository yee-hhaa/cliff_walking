# =============================================================================
# Cliff Walking：Q-learning 與 SARSA 比較（Sutton & Barto, Example 6.6, p.132）
# 【慢動作觀察版】可以一步一步看清楚 ε-greedy 在每一步做了什麼
# 作者：葉銘恩、黃昱維
#
#   * 軌跡箭頭：把這個回合走過的每一步留在畫面上
#       藍色 = 利用（exploit），選了 Q 值最大的動作
#       橙色 = 探索（explore），隨機選了一個動作
#   * 這一步選中的那個 Q 值會放大成紅色粗體，撞牆時畫一個空心圈
#   * 控制列：Q_learn／SARSA 按鈕、暫停／繼續、每步延遲滑桿（0 = 全速）
#
# 狀態編號：4 列 × 12 行 = 48 格，state = 列 × 12 + 行
#   行→   0   1   2   3   4   5   6   7   8   9  10  11
#   列0   0   1   2   3   4   5   6   7   8   9  10  11
#   列1  12  13  14  15  16  17  18  19  20  21  22  23
#   列2  24  25  26  27  28  29  30  31  32  33  34  35
#   列3  36  37  38  39  40  41  42  43  44  45  46  47   ← 36 起點、37~46 懸崖、47 終點
#
# 公式符號 ↔ 程式名稱：S = state、A = action、R = reward[next_state]、S' = next_state、
#   A' = next_action（只有 SARSA）、Q = ActionValue、α = Alpha、γ = Gamma、ε = Epsilon
# =============================================================================
import tkinter as tk  # 匯入 tkinter 圖形介面套件並取別名 tk，用來建立視窗、畫布、按鈕、滑桿與文字標籤
import numpy as np  # 匯入 numpy 並取別名 np，用來建立與更新 Q 表(48x4 的二維陣列)及產生亂數
import time  # 匯入 time 模組，慢動作版本真的會用到：time.time() 計算等待截止時間、time.sleep() 短暫休息

ARROW = ['上', '左', '下', '右']  # 動作編號 0~3 對應的中文方向，索引即動作編號，用來在狀態列顯示「選擇：上」之類的文字
EXPLOIT_COLOR = '#1e6fd9'  # 「利用」(選目前 Q 值最大的動作)時使用的顏色：藍色，用於箭頭、代理人圓點與狀態列文字
EXPLORE_COLOR = '#ff8c00'  # 「探索」(隨機選動作)時使用的顏色：橙色，讓使用者一眼看出這一步是不是隨機亂走
paused = False  # 全域旗標：目前是否處於暫停狀態，False 代表正常執行；由暫停按鈕切換

## define function
def TransMat(now_state, action):  # 狀態轉移函式：輸入目前狀態編號與動作，回傳執行動作後的下一個狀態編號(與原版完全相同)
    max_row = 4  # 地圖總列數 4 列
    max_col = 12  # 地圖總行數 12 行，共 4*12=48 個狀態，編號 0~47
    now_row = int(now_state/max_col)  # 狀態編號換算成列索引(0~3)，例如 36/12 = 3
    now_col = (now_state%max_col)  # 狀態編號換算成行索引(0~11)，例如 36%12 = 0

    if max_col < now_col or max_row < now_row or now_col < 0 or now_row < 0:  # 邊界檢查：行列超出範圍就視為錯誤(嚴格來說應該用 <=，這裡條件偏寬鬆，但正常流程不會觸發)
        print('index error')  # 印出索引錯誤訊息
        return  # 回傳 None 中止計算

    col = now_col  # 下一個位置的行先設為目前的行，撞牆時就維持原地
    row = now_row  # 下一個位置的列先設為目前的列，撞牆時就維持原地
    if action == 0 and now_row > 0:  # 動作 0「上」，且不在最上面一列才能移動
        row -= 1  # 列索引減 1，往上一格
    elif action == 1 and now_col > 0:  # 動作 1「左」，且不在最左邊一行才能移動
        col -= 1  # 行索引減 1，往左一格
    elif action == 2 and (max_row-1) > now_row:  # 動作 2「下」，且不在最下面一列才能移動
        row += 1  # 列索引加 1，往下一格
    elif action == 3 and (max_col-1) > now_col:  # 動作 3「右」，且不在最右邊一行才能移動
        col += 1  # 行索引加 1，往右一格
    next_state = row*max_col + col  # 把新的(列, 行)換算回狀態編號：列*12 + 行
    return next_state  # 回傳下一個狀態編號

OFFSET = [(50, 25), (25, 50), (50, 75), (75, 50)]  # 上、左、下、右四個 Q 值文字在一格(100x100)內的位置偏移 (dx, dy)，索引即動作編號；取代原版四段重複的 if/elif

def draw_q(state, action, value, highlight):  # 重畫某格某方向的 Q 值文字；highlight=True 時用紅色粗體放大強調，False 時用黑色一般字
    tag = str(state)+'_'+str(action)  # 組出這個文字物件的標籤，例如狀態 36 的動作 3 → '36_3'，用來刪除或更新
    dx, dy = OFFSET[action]  # 依動作編號取出該方向在格子內的偏移量
    mycanvas.delete(tag)  # 先刪除畫布上同標籤的舊文字，避免新舊數字重疊
    if highlight:  # 若要強調顯示
        mycanvas.create_text(10+(state%12*100)+dx, 10+(state//12*100)+dy, fill='#f00', font=('Arial', 20, 'bold'), text=value, tag=tag)  # 以紅色 20 級粗體畫出 Q 值；座標 = 整體偏移 10 + 格子左上角(行*100, 列*100) + 方向偏移
    else:  # 若是一般顯示
        mycanvas.create_text(10+(state%12*100)+dx, 10+(state//12*100)+dy, fill='#000', text=value, tag=tag)  # 以黑色預設字體畫出 Q 值，恢復平常樣式

def cell_center(s):  # 工具函式：把狀態編號換算成該格中心點的像素座標 (x, y)
    return 10+(s%12)*100+50, 10+(s//12)*100+50  # x = 10 + 行*100 + 50，y = 10 + 列*100 + 50；例如狀態 36 → (60, 360)

def wait_step():  # 依滑桿設定的毫秒數等待；暫停時會一直停在這裡，期間持續刷新畫面讓按鈕與滑桿仍可操作
    end = time.time() + speed_scale.get()/1000  # 計算等待的截止時間：現在時間 + 滑桿值(毫秒)換算成秒
    while time.time() < end or paused:  # 還沒到截止時間，或正在暫停中，就繼續等(兩個條件都不成立才離開)
        root.update()  # 處理 tkinter 事件並重繪畫面；少了這行，等待期間視窗會卡死、按暫停也沒反應
        time.sleep(0.01)  # 每圈休息 0.01 秒，避免 while 迴圈空轉把 CPU 吃滿

def toggle_pause():  # 「暫停／繼續」按鈕的回呼函式：切換暫停狀態
    global paused  # 宣告要修改的是全域變數 paused，而不是在函式內建立一個同名區域變數
    paused = not paused  # 把 True/False 反轉
    pause_button.config(text='繼續' if paused else '暫停')  # 依目前狀態更新按鈕文字：暫停中顯示「繼續」，執行中顯示「暫停」

def reset_trail():  # 每回合開始時呼叫：清掉上一回合的軌跡，並把代理人圓點放回起點
    mycanvas.delete('trail')  # 刪除所有標籤為 'trail' 的物件(箭頭與撞牆圓圈)
    mycanvas.delete('agent')  # 刪除代表代理人的圓點
    x, y = cell_center(36)  # 取得起點(狀態 36)的格子中心座標
    mycanvas.create_oval(x-12, y-12, x+12, y+12, fill='#888', outline='', tag='agent')  # 在起點畫一個半徑 12 的灰色實心圓代表代理人，無外框

def show_step(state, next_state, action, explored, r, epsilon, step, value):  # 把一步的過程分兩段動畫呈現：先顯示「選了哪個方向、為什麼」，再顯示「移動到哪裡」
    color = EXPLORE_COLOR if explored else EXPLOIT_COLOR  # 依這一步是探索還是利用決定顏色(橙／藍)
    mode = '探索：隨機選動作' if explored else '利用：選最大 Q 的動作'  # 依探索或利用決定狀態列上的說明文字
    sign = '<' if explored else '≥'  # 顯示亂數與 ε 的比較符號：探索是 r < ε，利用是 r ≥ ε
    # 第一段：代理人還在原格，選中方向的數字放大標紅
    draw_q(state, action, value, True)  # 把目前格子被選中方向的 Q 值用紅色放大顯示(注意：value 是這一步「更新後」的 Q 值，因為呼叫前已先做完 ValueUpdate)
    step_label.config(text=f'第 {step+1} 步｜亂數 {r:.3f} {sign} ε={epsilon} → {mode}｜選擇：{ARROW[action]}', fg=color)  # 狀態列顯示第幾步、亂數值(3 位小數)與 ε 的比較、判斷結果與選擇的方向，文字顏色跟著探索/利用
    wait_step()  # 等待滑桿設定的時間，讓使用者看清楚這一步的決策
    # 第二段：數字恢復，畫箭頭並把代理人移到下一格
    draw_q(state, action, value, False)  # 把剛剛標紅的數字恢復成黑色一般字
    x1, y1 = cell_center(state)  # 取得目前格子的中心座標(箭頭起點)
    x2, y2 = cell_center(next_state)  # 取得下一格的中心座標(箭頭終點)
    if next_state == state:  # 撞牆原地不動，畫一個圈表示
        mycanvas.create_oval(x1-18, y1-18, x1+18, y1+18, outline=color, width=2, tag='trail')  # 在原格畫一個半徑 18、線寬 2 的空心圓，標籤為 trail，回合結束時會被清掉
    else:  # 有實際移動
        mycanvas.create_line(x1, y1, x2, y2, fill=color, width=3, arrow=tk.LAST, tag='trail')  # 從目前格中心畫一條線寬 3 的箭頭到下一格中心，arrow=tk.LAST 表示箭頭畫在終點端
    mycanvas.delete('agent')  # 刪除舊位置的代理人圓點
    mycanvas.create_oval(x2-12, y2-12, x2+12, y2+12, fill=color, outline='', tag='agent')  # 在下一格中心重新畫代理人圓點，顏色與這一步的探索/利用一致
    step_label.config(text=f'第 {step+1} 步｜往{ARROW[action]}移動' + ('，撞牆留在原地' if next_state == state else ''), fg=color)  # 狀態列改為顯示移動方向；若撞牆則額外補上「撞牆留在原地」
    wait_step()  # 再等待一次，讓使用者看清楚移動結果

def qlearn(action_value, reward, steps, gamma, alpha, epsilon):  # Q-learning 演算法：跑完一個回合，回傳更新後的 Q 表、回合累積報酬、是否掉崖
    record = []  # 記錄每一步的 [狀態, 動作, 獎勵, 下一狀態]
    state = 36  # 起始狀態 36(左下角起點)
    fall = 0                # 記錄本回合是否掉入懸崖：0 = 沒掉，1 = 掉了
    reset_trail()  # 清空上一回合軌跡，把代理人放回起點
    for step in range(steps):  # 最多走 steps 步(1000 步)
        action, explored, r = GetAction(action_value, epsilon, state)  # 用 ε-greedy 選動作；新版 GetAction 會多回傳「是否探索」與「亂數值」供畫面顯示
        next_state = TransMat(state, action)  # 計算執行動作後的下一個狀態
        record.append([state, action, reward[next_state], next_state])  # 存下這一步的經驗 (S, A, R, S')
        action_value[state, action] = ValueUpdate('qlearn', action_value, record[step], alpha, gamma)  # 用 Q-learning 公式更新 Q(S, A) 並寫回 Q 表

        a2 = np.around(action_value, decimals=2)  # 產生四捨五入到小數 2 位的 Q 表副本，供畫面顯示

        show_step(state, next_state, action, explored, r, epsilon, step, a2[state, action])  # 以兩段動畫顯示這一步：先標紅選擇的方向並說明理由，再畫箭頭移動代理人

        state = next_state  # 推進到下一個狀態
        if state == 47:  # 走到終點 47
            print(",[success]")  # 印出成功訊息
            break  # 結束本回合
        elif state > 36:  # 掉進 37~46 的懸崖區
            print(",[fail]")  # 印出失敗訊息
            fall = 1            # 掉入懸崖，旗標設為 1
            break  # 結束本回合

    record = np.array(record, dtype=object)  # 轉成 object 型別的 numpy 陣列以便切片取欄位
    epi_reward = np.sum(record[:,2])  # 加總第 2 欄(獎勵)，得到本回合累積報酬
    return action_value, epi_reward, fall       # 比原版多回傳 fall，讓 main 能累計掉崖次數

def sarsa(action_value, reward, steps, gamma, alpha, epsilon):  # SARSA 演算法(on-policy)：跑完一個回合，回傳 Q 表、累積報酬、是否掉崖
    record = []  # 記錄每一步的 [狀態, 動作, 獎勵, 下一狀態, 下一動作]
    state = 36  # 起始狀態 36
    fall = 0                # 記錄本回合是否掉入懸崖
    reset_trail()  # 清空上一回合軌跡，代理人回到起點
    action, explored, r = GetAction(action_value, epsilon, state)  # SARSA 在迴圈前先選好第一個動作，同時記下它是不是探索、用的亂數是多少
    for step in range(steps):  # 最多走 steps 步
        next_state = TransMat(state, action)  # 執行目前動作，得到下一狀態
        next_action, next_explored, next_r = GetAction(action_value, epsilon, next_state)  # 在下一狀態先選好下一個動作 A'，並記下這次選擇是否探索與亂數值(下一步顯示時要用)
        record.append([state, action, reward[next_state], next_state, next_action])  # 存下 (S, A, R, S', A') 五元組
        action_value[state, action] = ValueUpdate('sarsa', action_value, record[step], alpha, gamma)  # 用 SARSA 公式更新 Q(S, A)

        a2 = np.around(action_value, decimals=2)  # 產生四捨五入到小數 2 位的 Q 表副本供顯示

        show_step(state, next_state, action, explored, r, epsilon, step, a2[state, action])  # 以兩段動畫顯示這一步；explored 與 r 是「當初選出 action 時」的判斷結果

        state = next_state  # 推進到下一狀態
        action = next_action  # 下一步實際執行剛剛選好的 A'(on-policy 的關鍵)
        explored, r = next_explored, next_r  # 同步把 A' 的探索判斷與亂數帶到下一步，讓下一步畫面顯示的理由與實際動作一致
        if state == 47:  # 抵達終點
            print(",[success]")  # 印出成功訊息
            break  # 結束本回合
        elif state > 36:  # 掉入懸崖區
            print(",[fail]")  # 印出失敗訊息
            fall = 1            # 掉入懸崖，旗標設為 1
            break  # 結束本回合

    record = np.array(record, dtype=object)  # 轉成 object 型別 numpy 陣列
    epi_reward = np.sum(record[:,2])  # 加總獎勵欄得到本回合累積報酬
    return action_value, epi_reward, fall       # 多回傳 fall

def GetAction(action_value, epsilon, next_state):  # ε-greedy 動作選擇：改為回傳 (動作, 是否探索, 亂數值) 三個值，方便畫面解釋每一步的決策
    r = np.random.rand()  # 產生一個 0~1 的純量亂數(原版是 rand(1) 回傳長度 1 的陣列，改成純量才方便用 f-string 格式化顯示)
    if r >= epsilon:  # 亂數 ≥ ε(機率 90%)：走「利用」路線
        action = int(np.argmax(action_value[next_state]))  # 只取這個狀態那一列的 4 個 Q 值找最大者的索引；效果與原版相同但更省計算；int() 轉成 Python 整數
        explored = False  # 標記這次不是探索
    else:  # 亂數 < ε(機率 10%)：走「探索」路線
        action = np.random.randint(0, 4)  # 隨機產生 0~3 的整數動作(純量)
        explored = True  # 標記這次是探索
    return action, explored, r  # 回傳動作、是否探索、所用的亂數值

def ValueUpdate(method, action_value, record, alpha, gamma):  # Q 值更新函式：依 qlearn 或 sarsa 計算新的 Q(S, A)(與原版相同)
    state = record[0]  # 取出目前狀態 S
    action = record[1]  # 取出執行的動作 A
    reward = record[2]  # 取出立即獎勵 R
    next_state = record[3]  # 取出下一狀態 S'
    now_value = action_value[state, action]  # 讀出舊的估計值 Q(S, A)
    if method == 'qlearn':  # Q-learning(off-policy)
        update_value = alpha*(reward + gamma*np.max(action_value[next_state,:]) - now_value)  # 增量 = α ×(R + γ × max Q(S', ·) − Q(S, A))，用下一狀態的「最佳」動作估計未來
    elif method == 'sarsa':  # SARSA(on-policy)
        next_action = record[4]  # 取出下一步實際要執行的動作 A'
        update_value = alpha*(reward + gamma*action_value[next_state, next_action] - now_value)  # 增量 = α ×(R + γ × Q(S', A') − Q(S, A))，用實際會執行的動作估計未來，因此會把探索的風險學進去
    else:  # 方法名稱不支援
        print('No this method.')  # 印出錯誤訊息
        return  # 回傳 None
    value = now_value + update_value  # 新 Q 值 = 舊值 + 增量(TD 更新)
    return value  # 回傳新 Q 值

# main
def main(episodes, method):  # 主訓練流程：設定環境與超參數，重複執行指定回合數，並即時更新掉崖統計
    ActionValue = np.zeros([48, 4])  # 建立 48x4 的 Q 表，全部初始化為 0
    Reward = np.full(48, -1)  # 每一格的獎勵預設為 -1(每走一步扣 1 分)
    Reward[37:-1] = -100  # 狀態 37~46(懸崖)的獎勵設為 -100
    EpisodeReward = []  # 收集每回合累積報酬的清單
    FallCount = 0               # 累計掉入懸崖的回合數

    Gamma = 0.99  # 折扣因子 γ
    Epsilon = 0.1  # 探索率 ε：10% 機率隨機選動作
    Steps = 1000  # 每回合最多 1000 步
    Alpha = 0.05  # 學習率 α

    if method == 'qlearn':  # 使用 Q-learning
        for episode in range(episodes):  # 重複 episodes 個回合
            print('now_episode = ', episode, end="")  # 印出目前回合編號，不換行，讓 [success]/[fail] 接在後面
            ActionValue, Epi_Reward, fall = qlearn(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)  # 執行一回合 Q-learning，接回 Q 表、累積報酬與是否掉崖
            EpisodeReward.append(Epi_Reward)  # 記錄本回合報酬
            FallCount += fall  # 若本回合掉崖(fall=1)就累加 1
            info_label.config(text=f'掉入懸崖次數：{FallCount} / {episode+1} 回合')  # 更新畫面下方的紅字統計：累計掉崖次數 / 已完成回合數
            root.update()  # 立即刷新畫面，讓統計數字馬上顯示(延遲設為 0 時尤其需要)
    elif method == 'sarsa':  # 使用 SARSA
        for episode in range(episodes):  # 重複 episodes 個回合
            print('now_episode = ', episode, end="")  # 印出目前回合編號，不換行
            ActionValue, Epi_Reward, fall = sarsa(ActionValue, Reward, Steps, Gamma, Alpha, Epsilon)  # 執行一回合 SARSA
            EpisodeReward.append(Epi_Reward)  # 記錄本回合報酬
            FallCount += fall  # 累加掉崖次數
            info_label.config(text=f'掉入懸崖次數：{FallCount} / {episode+1} 回合')  # 更新掉崖統計文字
            root.update()  # 立即刷新畫面
    else:  # 方法名稱不合法
        print('No this method.')  # 印出錯誤訊息
        return  # 中止函式

    EpisodeReward = np.array(EpisodeReward)  # 把回合報酬轉成 numpy 陣列(本程式尚未使用，可用來畫學習曲線)
    print('\n掉入懸崖總次數 =', FallCount, '/', episodes)  # 訓練結束後在終端機印出總掉崖次數；開頭 \n 是為了跟最後一行 [success]/[fail] 分開
    print(ActionValue)  # 印出訓練完成的完整 Q 表
    return ActionValue  # 回傳 Q 表供 drawline 畫出最佳路徑

def draw_map():  # 畫地圖：4x12 格線、起點與終點(與原版相同)
    mycanvas.delete("all")  # 清空畫布上所有物件
    mycanvas.create_rectangle(10, 10, 1210, 410)  # 畫地圖外框，1200x400 像素

    for i in range(110, 410, 100):  # y = 110、210、310
        mycanvas.create_line(10, i, 1210, i)  # 畫水平分隔線，分成 4 列

    for i in range(110, 1210, 100):  # x = 110 ~ 1110
        mycanvas.create_line(i, 10, i, 410)  # 畫垂直分隔線，分成 12 行

    mycanvas.create_rectangle(10, 310, 110, 410, fill='#ccffcd')  # 左下角起點(狀態 36)塗淡綠色
    mycanvas.create_rectangle(1110, 310, 1210, 410, fill='#ccf')  # 右下角終點(狀態 47)塗淡藍色

def drawline(reward):  # 訓練完成後：依 Q 表從起點沿貪婪策略畫出粉紅色最佳路徑，並重新標上所有 Q 值(參數名叫 reward，實際傳入的是 Q 表)
    mycanvas.delete('trail')  # 清掉最後一回合留下的箭頭與圓圈軌跡，避免和最終路徑混在一起
    mycanvas.delete('agent')  # 移除代理人圓點
    step_label.config(text='訓練完成', fg='#000')  # 狀態列改成黑色的「訓練完成」
    for i in range(4):  # 走訪 4 列
        for j in range(12):  # 走訪 12 行
            mycanvas.delete(str(j+(12*i))+'_0')  # 刪除該格上方的 Q 值文字
            mycanvas.delete(str(j+(12*i))+'_1')  # 刪除該格左方的 Q 值文字
            mycanvas.delete(str(j+(12*i))+'_2')  # 刪除該格下方的 Q 值文字
            mycanvas.delete(str(j+(12*i))+'_3')  # 刪除該格右方的 Q 值文字

    now_state = 36  # 從起點開始追蹤路徑
    i, j = 60, 360  # 畫線起點座標，即起點格中心
    while (now_state != 47):  # 還沒到終點就繼續(若策略形成迴圈會變成無窮迴圈，視窗卡住)
        m = max(reward[now_state])  # 取目前狀態 4 個動作中最大的 Q 值
        if m == reward[now_state][0]:  # 最大值是「上」
            mycanvas.create_line(i, j, i, j-100, fill='#fcc', width=5)  # 往上畫粉紅色線段
            j = j-100  # y 座標往上一格
            now_state = now_state-12  # 狀態往上一列(減 12)
        elif m == reward[now_state][1]:  # 最大值是「左」
            mycanvas.create_line(i, j, i-100, j, fill='#fcc', width=5)  # 往左畫粉紅色線段
            i = i-100  # x 座標往左一格
            now_state = now_state-1  # 狀態往左一行(減 1)
        elif m == reward[now_state][2]:  # 最大值是「下」
            mycanvas.create_line(i, j, i, j+100, fill='#fcc', width=5)  # 往下畫粉紅色線段
            j = j+100  # y 座標往下一格
            now_state = now_state+12  # 狀態往下一列(加 12)
        elif m == reward[now_state][3]:  # 最大值是「右」
            mycanvas.create_line(i, j, i+100, j, fill='#fcc', width=5)  # 往右畫粉紅色線段
            i = i+100  # x 座標往右一格
            now_state = now_state+1  # 狀態往右一行(加 1)

    for i in range(4):  # 路徑畫完後走訪 4 列
        for j in range(12):  # 走訪 12 行，重新標上最終 Q 值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(reward[j+(12*i)][0],'.2f'), tag=str(j+(12*i))+'_0')  # 格子上方標「上」的 Q 值，小數 2 位
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(reward[j+(12*i)][1],'.2f'), tag=str(j+(12*i))+'_1')  # 格子左方標「左」的 Q 值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(reward[j+(12*i)][2],'.2f'), tag=str(j+(12*i))+'_2')  # 格子下方標「下」的 Q 值
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(reward[j+(12*i)][3],'.2f'), tag=str(j+(12*i))+'_3')  # 格子右方標「右」的 Q 值

    root.update()  # 刷新視窗，讓路徑與數值立刻顯示


def Q_learn():  # 「Q_learn」按鈕回呼：重畫地圖、重置統計、標上初始 0.00，然後開始 Q-learning 訓練
    draw_map()  # 重畫地圖，清除上一次的內容
    info_label.config(text='掉入懸崖次數：0')  # 把掉崖統計歸零
    ActionValue = np.zeros([48, 4])  # 全 0 的 Q 表，只用來畫初始畫面的 0.00(真正訓練用的在 main 裡)

    for i in range(4):  # 走訪 4 列
        for j in range(12):  # 走訪 12 行
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(ActionValue[j+(12*i)][0],'.2f'), tag=str(j+(12*i))+'_0')  # 標出「上」的初始值並掛標籤
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][1],'.2f'), tag=str(j+(12*i))+'_1')  # 標出「左」的初始值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(ActionValue[j+(12*i)][2],'.2f'), tag=str(j+(12*i))+'_2')  # 標出「下」的初始值
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][3],'.2f'), tag=str(j+(12*i))+'_3')  # 標出「右」的初始值

    q_reward = main(1000, 'qlearn')  # 以 Q-learning 訓練 1000 回合，取回 Q 表
    drawline(q_reward)  # 畫出學到的貪婪路徑(通常緊貼懸崖)

def SARSA():  # 「SARSA」按鈕回呼：流程同上，改用 SARSA 訓練
    draw_map()  # 重畫地圖
    info_label.config(text='掉入懸崖次數：0')  # 掉崖統計歸零
    ActionValue = np.zeros([48, 4])  # 只供初始畫面顯示用的全 0 Q 表

    for i in range(4):  # 走訪 4 列
        for j in range(12):  # 走訪 12 行
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+25, text=format(ActionValue[j+(12*i)][0],'.2f'), tag=str(j+(12*i))+'_0')  # 標出「上」的初始值
            mycanvas.create_text(10+(j*100)+25, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][1],'.2f'), tag=str(j+(12*i))+'_1')  # 標出「左」的初始值
            mycanvas.create_text(10+(j*100)+50, 10+(i*100)+75, text=format(ActionValue[j+(12*i)][2],'.2f'), tag=str(j+(12*i))+'_2')  # 標出「下」的初始值
            mycanvas.create_text(10+(j*100)+75, 10+(i*100)+50, text=format(ActionValue[j+(12*i)][3],'.2f'), tag=str(j+(12*i))+'_3')  # 標出「右」的初始值

    s_reward = main(1000, 'sarsa')  # 以 SARSA 訓練 1000 回合
    drawline(s_reward)  # 畫出學到的路徑(通常離懸崖較遠)


root = tk.Tk()  # 建立 tkinter 主視窗
root.title('Cliff_Walking')  # 設定視窗標題
mycanvas = tk.Canvas(root, width=1220, height=420)  # 建立 1220x420 的畫布，用來畫地圖、Q 值、軌跡與代理人
mycanvas.pack()  # 把畫布放進視窗

draw_map()  # 啟動時先畫出空白地圖

ctrl = tk.Frame(root)  # 建立一個框架容器，用來把按鈕與滑桿橫向排成一列
ctrl.pack(pady=5)  # 放進視窗，上下各留 5 像素間距
tk.Button(ctrl, text='Q_learn', command=Q_learn).pack(side=tk.LEFT, padx=5)  # 建立 Q_learn 按鈕並直接靠左排列(不存變數，因為之後不需要再改它)
tk.Button(ctrl, text='SARSA', command=SARSA).pack(side=tk.LEFT, padx=5)  # 建立 SARSA 按鈕並靠左排列在 Q_learn 右邊
pause_button = tk.Button(ctrl, text='暫停', width=6, command=toggle_pause)  # 建立暫停按鈕並存進變數 pause_button，因為 toggle_pause 要修改它的文字；width=6 固定寬度，切換文字時按鈕不會跳動
pause_button.pack(side=tk.LEFT, padx=5)  # 暫停按鈕靠左排列(pack 要分開寫，若寫成 Button(...).pack() 變數會拿到 None)
speed_scale = tk.Scale(ctrl, from_=0, to=1000, resolution=10, orient=tk.HORIZONTAL, length=300, label='每步延遲（毫秒，0 = 全速）')  # 建立水平滑桿：範圍 0~1000 毫秒、每格 10 毫秒、長 300 像素，控制每段動畫的等待時間
speed_scale.set(300)  # 預設每段 0.3 秒(每步分兩段，所以一步約 0.6 秒)
speed_scale.pack(side=tk.LEFT, padx=10)  # 滑桿靠左排列在按鈕右邊

legend = tk.Frame(root)  # 建立顏色圖例用的框架
legend.pack()  # 放在控制列下方
tk.Label(legend, text='● 利用（亂數 ≥ ε，選最大 Q）', fg=EXPLOIT_COLOR, font=('Arial', 12)).pack(side=tk.LEFT, padx=10)  # 藍色圖例文字，說明藍色代表利用
tk.Label(legend, text='● 探索（亂數 < ε，隨機選）', fg=EXPLORE_COLOR, font=('Arial', 12)).pack(side=tk.LEFT, padx=10)  # 橙色圖例文字，說明橙色代表探索

step_label = tk.Label(root, text='按下 Q_learn 或 SARSA 開始', font=('Arial', 13))  # 狀態列標籤：訓練時由 show_step 更新，顯示每一步的 ε-greedy 判斷過程
step_label.pack(pady=3)  # 放進視窗，上下留 3 像素

info_label = tk.Label(root, text='掉入懸崖次數：0', font=('Arial', 14), fg='red')  # 紅色統計標籤：顯示累計掉崖次數 / 回合數
info_label.pack(pady=5)  # 放進視窗，上下留 5 像素

author_label = tk.Label(root, text='作者：葉銘恩、黃昱維', font=('Arial', 12), fg='#555')  # 作者標籤：以灰色 12 級字顯示作者姓名
author_label.pack(pady=(0, 8))  # 放在視窗最下方，上方不留距、下方留 8 像素

root.mainloop()  # 啟動 tkinter 事件主迴圈，等待使用者操作直到關閉視窗
