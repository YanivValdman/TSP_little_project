#!/usr/bin/env python3
"""
Quick test for the menu with shorter timeout
"""
import pygame
import sys
import time

def test_menu():
    pygame.init()
    screen = pygame.display.set_mode((600, 300))
    pygame.display.set_caption("Menu Test")
    font = pygame.font.SysFont(None, 50)
    small_font = pygame.font.SysFont(None, 28)
    
    options = ["Use default configuration", "Use custom number of points"]
    selected = 0
    menu_active = True
    timeout = 3  # 3 seconds for testing
    menu_start_time = time.time()
    
    BUTTON_COLOR = (180, 180, 180)
    BUTTON_COLOR_ACTIVE = (90, 180, 90)
    
    def draw_wrapped_text(text, font, color, x, y, max_width, surface):
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            test_surface = font.render(test_line, True, color)
            if test_surface.get_width() <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        for line in lines:
            line_surface = font.render(line, True, color)
            surface.blit(line_surface, (x, y))
            y += line_surface.get_height() + 2
        return y
    
    clock = pygame.time.Clock()
    
    while menu_active:
        screen.fill((240, 240, 240))
        elapsed = time.time() - menu_start_time
        timeout_rem = timeout - int(elapsed)
        
        # Add title and instructions
        title = "TSP Tracker Startup"
        title_surface = font.render(title, True, (50, 50, 150))
        screen.blit(title_surface, (50, 20))
        
        instructions = "Use UP/DOWN arrow keys to navigate, ENTER to select"
        instructions_surface = small_font.render(instructions, True, (80, 80, 80))
        screen.blit(instructions_surface, (50, 60))
        
        for i, option in enumerate(options):
            color = (0, 0, 0)
            bg = BUTTON_COLOR_ACTIVE if i == selected else BUTTON_COLOR
            rect = pygame.Rect(50, 100 + i*70, 500, 60)
            pygame.draw.rect(screen, bg, rect)
            text = font.render(option, True, color)
            screen.blit(text, (rect.x + 15, rect.y + 10))
            
        # Timeout info
        info = f"Auto-selecting default in {timeout_rem}s..." if timeout_rem > 0 else "Auto-selected default."
        info_col = (100, 60, 60) if timeout_rem > 0 else (0, 120, 0)
        draw_wrapped_text(info, small_font, info_col, 50, 250, 500, screen)
        
        pygame.display.flip()
        clock.tick(30)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    print(f"Selected option {selected}: {options[selected]}")
                    pygame.quit()
                    return
                    
        if elapsed >= timeout:
            print("Timeout reached - auto-selecting default")
            pygame.quit()
            return
    
if __name__ == "__main__":
    print("Testing improved menu display...")
    test_menu()
    print("Menu test completed!")