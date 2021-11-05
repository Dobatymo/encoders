#include <string>

class PyUnicode {

public:
	int kind;
	std::string unicode;

	PyUnicode() {}
	PyUnicode(int kind, const std::string &unicode) : kind(kind), unicode(unicode) {}
	
	bool operator==(const PyUnicode &other) const noexcept {
		return kind == other.kind && unicode == other.unicode;
	}
};

template <> struct std::hash<PyUnicode> {
	std::size_t operator()(PyUnicode const &s) const noexcept {
		std::size_t h1 = std::hash<int>{}(s.kind);
		std::size_t h2 = std::hash<std::string>{}(s.unicode);
		return h1 ^ (h2 << 1);
	}
};