// From: Dr. Dobb's: A Highly Configurable Logging Framework In C++

#ifndef _LOGGER_HPP_
#define _LOGGER_HPP_

#include <stdexcept>
#include <iostream>
#include <sstream>

#define LOG_PARAMS_UNUSED(x) (void)x

namespace onnx {
namespace sd {
namespace amon {

enum loglevel_e {
  LOGGER_CRIT    = 1,
  LOGGER_ERR     = 2,
  LOGGER_WARN    = 3,
  LOGGER_INFO    = 4,
  LOGGER_DEBUG   = 5,
  LOGGER_VERBOSE = 6
};

class logIt {
private:
    std::ostringstream _buffer;

public:
    logIt(enum loglevel_e _loglevel = LOGGER_INFO) {
        _buffer << _loglevel << " :"
                << std::string(
                    _loglevel > LOGGER_VERBOSE
                    ? (_loglevel - LOGGER_VERBOSE) * 4
                    : 1
                    , ' ');
    }

    template <typename T>
    logIt & operator<<(T const & value)
    {
        _buffer << value;
        return *this;
    }

    ~logIt()
    {
        _buffer << std::endl;
        // This is atomic according to the POSIX standard
        // http://www.gnu.org/s/libc/manual/html_node/Streams-and-Threads.html
        std::cerr << _buffer.str();
    }
};

#define sd_log(level) logIt(level)

} // namespace amon
} // namespace sd
} // namespace onnx

#endif // _LOGGER_HPP_