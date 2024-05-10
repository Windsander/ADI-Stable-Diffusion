/*
 * EntityRegister 实体注册机基类
 * Definition: 通用注册机基类
 *             定义了基本的注册机方法，并提供统一实现
 * Created by Arikan.Li on 2022/03/11.
 */
#include "onnxsd_defs.h"

#ifndef EXPOSE_FUNCTION_MAP_AUTO
#define REG_EXPOSE_FUNCTION_MAP_AUTO()                                                                    \
private:                                                                                                  \
ExposeFuncMap* function_map;                                                                              \
                                                                                                          \
protected:                                                                                                \
void insert_method(const ExposeFunctionKey &key, const ExposeFunction &target_function)                   \
{                                                                                                         \
  function_map->emplace(key, target_function);                                                            \
}                                                                                                         \
                                                                                                          \
public:                                                                                                   \
void* call_method(const std::string &key, void* params)                                                   \
{                                                                                                         \
  if (function_map->find(key) == function_map->end()) {                                                   \
    render_report(register_exception(EXC_LOG_ERR, "ERROR:: Calling method not exist!"));                  \
  }                                                                                                       \
  return (*function_map)[key](params);                                                                    \
}

#define ADD_EXPOSE_FUNCTION_MAP_AUTO()                                                                    \
function_map = new ExposeFuncMap()

#define DEL_EXPOSE_FUNCTION_MAP_AUTO()                                                                    \
function_map->clear();                                                                                    \
delete function_map
#endif // EXPOSE_FUNCTION_MAP_AUTO

namespace onnx {
namespace sd {
namespace base {
using namespace amon;

/*实体配置表=============================================================*/
/* 配置表结构体 */
class EntityConfig {
public:
  typedef std::map<std::string, void*> config_map;

private:
  config_map res_config_map;

public:
  EntityConfig() = default;

  explicit EntityConfig(const config_map &_res_config)
  {
    res_config_map = _res_config;
  }

  // Copy constructor.
  EntityConfig(const EntityConfig &income)
  {
    if (this != &income) {
      res_config_map = income.res_config_map;
    }
  }

  // Move constructor.
  EntityConfig(EntityConfig &&income) noexcept
  {
    if (this != &income) {
      res_config_map = income.res_config_map;
    }
  }

  EntityConfig &operator=(const EntityConfig &income)
  {
    if (this != &income) {
      res_config_map = income.res_config_map;
    }
    return *this;
  };

  ~EntityConfig()
  {
    res_config_map.clear();
  }

public:
  /* 获取全部配置值 */
  config_map
  get_all_configs()
  {
    if (res_config_map.empty()) {
      render_report(register_exception(EXC_LOG_DEBUG, "ERROR:: config is NULL! That's unsafe!"));
    }
    return res_config_map;
  }

  /* 获取指定配置值 */
  void*
  get_certain_config(const std::string &config_name)
  {
    config_map::iterator config_finder;
    config_finder = res_config_map.find(config_name);
    if (config_finder == res_config_map.end()) {
      render_report(register_exception(EXC_LOG_DEBUG, "ERROR:: certain config not find!"));
    } else {
      return config_finder->second;
    }
    return nullptr;
  }
};

/*管理器定义=============================================================*/
/* 资源包注册体 */
struct EntityRegisterPack {
  std::function<void*()>                          create_method;
  std::function<void*(void*)>                     release_method;
  std::function<void*(void*, std::string, void*)> call_method;

  explicit EntityRegisterPack(std::function<void*()> _create_method,
                              std::function<void*(void*)> _release_method,
                              std::function<void*(void*, std::string, void*)> _call_method)
  {
    create_method  = _create_method;
    release_method = _release_method;
    call_method    = _call_method;
  }

  ~EntityRegisterPack()
  {
    create_method  = nullptr;
    release_method = nullptr;
    call_method    = nullptr;
  }
};

/*注册自动宏=============================================================*/

template<typename target_register, typename target_entity, typename target_type = int>
class EntityRegister {
  typedef target_register Register;
  typedef target_entity   Entity;
  typedef target_type     EntityType;

private:
  typedef std::map<EntityType, EntityRegisterPack> GeneratorMap;

  GeneratorMap map_;
  bool         current_register_started;

protected:
  static Register* instance;
  static volatile uint32_t occupation_counter;

  EntityRegister()
  {
    current_register_started = true;
  }

  virtual ~EntityRegister()
  {
    current_register_started = false;
  }

  /* 注册入口 */
  template<typename Implement>
  void signin_generator(EntityType key)
  {
    EntityRegister::map_.emplace(key, EntityRegisterPack(
        []() -> void* {
          Implement* object = new Implement();
          return object;
        },
        [](void* object) -> void* {
          delete ((Implement*) object);
          return nullptr;
        },
        [](void* object, const std::string &method_name, void* params) -> void* {
          return ((Implement*) object)->call_method(method_name, params);
        }
    ));
  }

public:
  /* 生成器目录 */
  virtual void regist_all_generator() = 0;
  void unsign_all_generator()
  {
    map_.clear();
  }

  /* 启停控制 */
  static bool manual_init()
  {
    occupation_counter++;
    if (nullptr == instance && occupation_counter > 0) {
      instance = new Register();
      instance->regist_all_generator();
    }
    render_report(register_exception(EXC_LOG_INFO, "EntityRegister::manager_init()"));
    return EntityRegister::is_register_running();
  }

  static void manual_destroy()
  {
    occupation_counter--;
    if (nullptr != instance && occupation_counter <= 0) {
      instance->unsign_all_generator();
      delete instance;
      instance = nullptr;
    }
  }

  static bool is_register_running()
  {
    render_report(register_exception(EXC_LOG_INFO, "EntityRegister::check_register_running()"));
    return instance->current_register_started;
  }

  /* 操作入口 */
  static Entity* generate_entity(EntityType key, Entity** entity_p = nullptr)
  {
    assert(instance);
    if (instance->map_.find(key) == instance->map_.end()) {
      render_report(register_exception(EXC_LOG_ERR, "ERROR:: aim Entity Implement generate method not registered!"));
    }
    if (!entity_p) {
      return (Entity *)(instance->map_.find(key)->second.create_method());
    }
    return (*entity_p = (Entity*) (instance->map_.find(key)->second.create_method()));
  }

  static Entity* release_entity(EntityType key, Entity* entity)
  {
    assert(instance);
    if (instance->map_.find(key) == instance->map_.end()) {
      render_report(register_exception(EXC_LOG_ERR, "ERROR:: aim Entity Implement generate method not registered!"));
    }
    return (Entity*) (instance->map_.find(key)->second.release_method(entity));
  }

  static void* call_method(Entity* object, const std::string &method_name, void* params)
  {
    assert(instance);
    const EntityType key = object->get_driver_type();
    if (instance->map_.find(key) == instance->map_.end()) {
      render_report(register_exception(EXC_LOG_ERR, "ERROR:: aim Entity Implement calling method not registered!"));
    }
    return instance->map_.find(key)->second.call_method(object, method_name, params);
  }
};

/*注册与管理处理逻辑==================================================*/

template<typename target_register, typename target_entity, typename target_type>
target_register* EntityRegister<target_register, target_entity, target_type>::instance;

template<typename target_register, typename target_entity, typename target_type>
volatile uint32_t EntityRegister<target_register, target_entity, target_type>::occupation_counter = 0;

} // namespace base
} // namespace sd
} // namespace onnx
