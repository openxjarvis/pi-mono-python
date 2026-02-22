import curses
import random

def main(stdscr):
    # 初始化 curses 设置
    curses.curs_set(0)
    stdscr.nodelay(1)
    stdscr.timeout(100)

    sh, sw = stdscr.getmaxyx()
    
    # 初始蛇的位置 (y, x)
    # 蛇身初始为3节，水平排列
    snk_x = sw // 4
    snk_y = sh // 2
    snake = [
        [snk_y, snk_x],
        [snk_y, snk_x - 1],
        [snk_y, snk_x - 2]
    ]

    # 初始食物位置
    food = [sh // 2, sw // 2]
    try:
        stdscr.addch(food[0], food[1], '*')
    except curses.error:
        pass

    # 初始按键方向
    key = curses.KEY_RIGHT

    while True:
        next_key = stdscr.getch()
        
        # 简单的防掉头逻辑
        if next_key == -1:
            key = key
        elif next_key == curses.KEY_DOWN and key != curses.KEY_UP:
            key = next_key
        elif next_key == curses.KEY_UP and key != curses.KEY_DOWN:
            key = next_key
        elif next_key == curses.KEY_LEFT and key != curses.KEY_RIGHT:
            key = next_key
        elif next_key == curses.KEY_RIGHT and key != curses.KEY_LEFT:
            key = next_key
        # ESC退出
        elif next_key == 27:
            break

        # 计算新蛇头位置
        new_head = [snake[0][0], snake[0][1]]

        if key == curses.KEY_DOWN:
            new_head[0] += 1
        elif key == curses.KEY_UP:
            new_head[0] -= 1
        elif key == curses.KEY_LEFT:
            new_head[1] -= 1
        elif key == curses.KEY_RIGHT:
            new_head[1] += 1

        # 检查是否撞墙
        if (new_head[0] in [0, sh] or 
            new_head[1] in [0, sw] or 
            new_head in snake):
            break

        # 插入新蛇头
        snake.insert(0, new_head)

        # 检查是否吃到食物
        if snake[0] == food:
            food = None
            while food is None:
                nf = [
                    random.randint(1, sh - 2),
                    random.randint(1, sw - 2)
                ]
                if nf not in snake:
                    food = nf
            try:
                stdscr.addch(food[0], food[1], '*')
            except curses.error:
                pass
        else:
            # 没吃到，移除尾部
            tail = snake.pop()
            try:
                stdscr.addch(tail[0], tail[1], ' ')
            except curses.error:
                pass

        # 绘制蛇头
        try:
            stdscr.addch(snake[0][0], snake[0][1], '#')
        except curses.error:
            pass

    # 游戏结束
    stdscr.timeout(-1)
    msg = f"Game Over! Score: {len(snake) - 3}"
    try:
        stdscr.addstr(sh // 2, (sw - len(msg)) // 2, msg)
        stdscr.addstr(sh // 2 + 1, (sw - 23) // 2, "Press any key to exit...")
    except curses.error:
        pass
    stdscr.getch()

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure your terminal window is large enough.")
