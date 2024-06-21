# Defines colorize CMake output
# code adapted from stackoverflow: http://stackoverflow.com/a/19578320
# Created:
#
# - by Arikan.Li on 2021/12/31.
#
# Note:
#
# - ???What?

# 资源归档=================================================================================================

macro(define_colors)
    if(WIN32)
        # has no effect on WIN32
        set(ColourReset "")
        set(ColourBold "")
        set(Red "")
        set(Green "")
        set(Yellow "")
        set(Blue "")
        set(Magenta "")
        set(Cyan "")
        set(White "")
        set(BoldRed "")
        set(BoldGreen "")
        set(BoldYellow "")
        set(BoldBlue "")
        set(BoldMagenta "")
        set(BoldCyan "")
        set(BoldWhite "")
    else()
        string(ASCII 27 Esc)
        set(ColourReset "${Esc}[m")
        set(ColourBold "${Esc}[1m")
        set(Red "${Esc}[31m")
        set(Green "${Esc}[32m")
        set(Yellow "${Esc}[33m")
        set(Blue "${Esc}[34m")
        set(Magenta "${Esc}[35m")
        set(Cyan "${Esc}[36m")
        set(White "${Esc}[37m")
        set(BoldRed "${Esc}[1;31m")
        set(BoldGreen "${Esc}[1;32m")
        set(BoldYellow "${Esc}[1;33m")
        set(BoldBlue "${Esc}[1;34m")
        set(BoldMagenta "${Esc}[1;35m")
        set(BoldCyan "${Esc}[1;36m")
        set(BoldWhite "${Esc}[1;37m")
    endif()
endmacro()

function(colorize_option value result_var)
    if("${value}" STREQUAL "ON")
        set(result "${Cyan}${value}${ColourReset}")
    elseif("${value}" STREQUAL "OFF")
        set(result "${Red}${value}${ColourReset}")
    else()
        set(result "${value}")
    endif()
    set(${result_var} "${result}")
endfunction()

define_colors()